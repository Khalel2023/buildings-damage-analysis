import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from pathlib import Path
import sys
import json
import shutil
import cv2
import tempfile
import numpy as np
from shapely import wkt
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Импорт скрипта для обработки изображений

from full_interference import BuildingDamagePipeline


class Args:
    def __init__(self, pre_image, post_image, output_dir, seg_weights, cls_weights):
        self.pre_image = pre_image
        self.post_image = post_image
        self.output_dir = output_dir
        self.seg_weights = seg_weights
        self.cls_weights = cls_weights
        self.seg_threshold = 0.5
        self.adaptive_threshold = True
        self.batch_size = 32
        self.debug = False


# Константы для путей к моделям
SEG_WEIGHTS = ''
CLS_WEIGHTS = ''
# Конфигурация приложения
app = Flask(__name__)
app.config.from_pyfile('config.py')

# Инициализация Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# База данных
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)


# Модель пользователя
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    organization = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Модель для хранения результатов анализа
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pre_image_path = db.Column(db.String(255))
    post_image_path = db.Column(db.String(255))
    result_json_path = db.Column(db.String(255))
    visualization_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_public = db.Column(db.Boolean, default=False)


# Инициализация базы данных
with app.app_context():
    db.create_all()
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Маршруты
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))

        if not username or not password:
            flash('Пожалуйста, введите имя пользователя и пароль', 'warning')
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            flash('Неверное имя пользователя или пароль', 'danger')
            return redirect(url_for('login'))

        if not user.is_active:
            flash('Аккаунт отключен', 'danger')
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name', '')
        organization = request.form.get('organization', '')

        if not username or not email or not password:
            flash('Пожалуйста, заполните все обязательные поля', 'warning')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Имя пользователя уже занято', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email уже используется', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            full_name=full_name,
            organization=organization
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Регистрация успешна! Теперь вы можете войти.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Вы успешно вышли из системы', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).limit(5).all()
    return render_template('dashboard.html', analyses=analyses)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        try:
            # 1. Проверка наличия файлов в запросе
            if 'pre_image' not in request.files or 'post_image' not in request.files:
                flash('Необходимо загрузить оба изображения', 'warning')
                return redirect(request.url)

            pre_image = request.files['pre_image']
            post_image = request.files['post_image']

            # 2. Валидация файлов
            if not pre_image.filename or not post_image.filename:
                flash('Не выбраны файлы для загрузки', 'warning')
                return redirect(request.url)

            # 3. Создание временной рабочей директории
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                try:
                    # 4. Сохранение изображений с проверкой
                    pre_path = temp_dir / "pre_image.jpg"
                    post_path = temp_dir / "post_image.jpg"

                    # Сохраняем через PIL с проверкой
                    try:
                        Image.open(pre_image).convert('RGB').save(pre_path, "JPEG", quality=95)
                        Image.open(post_image).convert('RGB').save(post_path, "JPEG", quality=95)
                    except Exception as e:
                        raise ValueError(f"Ошибка конвертации изображений: {str(e)}")

                    # Проверка через OpenCV
                    if cv2.imread(str(post_path)) is None:
                        raise ValueError("Не удалось прочитать изображение")

                    # 5. Создание постоянной директории для анализа
                    analysis_uuid = str(uuid.uuid4())
                    analysis_dir = upload_dir / analysis_uuid
                    analysis_dir.mkdir(parents=True, exist_ok=True)

                    # 6. Перенос файлов в постоянную директорию
                    final_pre_path = analysis_dir / "pre_image.jpg"
                    final_post_path = analysis_dir / "post_image.jpg"

                    shutil.move(str(pre_path), str(final_pre_path))
                    shutil.move(str(post_path), str(final_post_path))

                    # 7. Запуск анализа
                    args = Args(
                        pre_image=str(final_pre_path.absolute()),
                        post_image=str(final_post_path.absolute()),
                        output_dir=str(analysis_dir.absolute()),
                        seg_weights=SEG_WEIGHTS,
                        cls_weights=CLS_WEIGHTS
                    )

                    pipeline = BuildingDamagePipeline(args)
                    results = pipeline.run()

                    # 8. Проверка результатов анализа
                    if not Path(results['visualization']).exists():
                        raise RuntimeError("Файл визуализации не был создан")

                    # 9. Сохранение в БД
                    analysis = Analysis(
                        user_id=current_user.id,
                        pre_image_path=str(final_pre_path.relative_to(upload_dir)),
                        post_image_path=str(final_post_path.relative_to(upload_dir)),
                        result_json_path=str(Path(results['final_results']).relative_to(upload_dir)),
                        visualization_path=str(Path(results['visualization']).relative_to(upload_dir))
                    )

                    db.session.add(analysis)
                    db.session.commit()

                    flash('Анализ успешно завершен!', 'success')
                    return redirect(url_for('results', analysis_id=analysis.id))

                except Exception as e:
                    # Автоматическая очистка временных файлов через контекстный менеджер
                    logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
                    flash(f'Ошибка обработки: {str(e)}', 'danger')
                    return redirect(request.url)

        except Exception as e:
            logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
            flash('Произошла критическая ошибка при обработке запроса', 'danger')
            return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    try:
        analysis = db.session.get(Analysis, analysis_id)
        if not analysis:
            flash('Анализ не найден', 'danger')
            return redirect(url_for('dashboard'))

        if analysis.user_id != current_user.id:
            flash('Доступ запрещен', 'danger')
            return redirect(url_for('dashboard'))

        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        result_json_path = upload_dir / analysis.result_json_path

        if not result_json_path.exists():
            flash('Файл результатов не найден', 'danger')
            return redirect(url_for('dashboard'))

        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        buildings = []
        damage_stats = {
            'no-damage': 0,
            'minor-damage': 0,
            'major-damage': 0,
            'destroyed': 0
        }

        for feature in result_data.get('features', {}).get('xy', []):
            try:
                props = feature.get('properties', {})
                if not props.get('uid'):
                    continue

                damage = props.get('damage', 'no-damage')
                if damage not in damage_stats:
                    damage = 'no-damage'

                confidence = max(0, min(1, float(props.get('confidence', 0))))

                buildings.append({
                    'id': props['uid'],
                    'damage': damage,
                    'confidence': f"{confidence:.2f}",
                    'coordinates': feature.get('wkt', 'N/A')
                })
                damage_stats[damage] += 1

            except Exception as e:
                logger.warning(f"Ошибка обработки здания: {str(e)}")
                continue

        # Проверяем доступность визуализации
        visualization_path = upload_dir / analysis.visualization_path
        has_visualization = visualization_path.exists()

        return render_template(
            'results.html',
            analysis=analysis,
            buildings=buildings,
            damage_stats=damage_stats,
            damage_colors={
                'no-damage': '#32ff32',
                'minor-damage': '#ffff00',
                'major-damage': '#0000ff',
                'destroyed': '#ff0000'
            },
            total_buildings=len(buildings),
            has_visualization=has_visualization
        )

    except Exception as e:
        logger.error(f"Ошибка в маршруте results: {str(e)}", exc_info=True)
        flash('Внутренняя ошибка сервера', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/download/<int:analysis_id>/<file_type>')
@login_required
def download(analysis_id, file_type):
    try:
        # Получаем анализ из базы данных
        analysis = db.session.get(Analysis, analysis_id)
        if not analysis:
            flash('Анализ не найден', 'danger')
            return redirect(url_for('dashboard'))

        # Проверяем права доступа
        if analysis.user_id != current_user.id:
            flash('Доступ запрещен', 'danger')
            return redirect(url_for('dashboard'))

        # Определяем пути к файлам
        upload_dir = Path(app.config['UPLOAD_FOLDER'])

        # Создаем полный путь к файлу
        if file_type == 'visualization':
            if not analysis.visualization_path:
                flash('Файл визуализации не найден в базе данных', 'danger')
                return redirect(url_for('results', analysis_id=analysis_id))

            file_path = upload_dir / analysis.visualization_path
            mime_type = 'image/png'

        elif file_type == 'json':
            if not analysis.result_json_path:
                flash('JSON файл не найден в базе данных', 'danger')
                return redirect(url_for('results', analysis_id=analysis_id))

            file_path = upload_dir / analysis.result_json_path
            mime_type = 'application/json'

        elif file_type == 'pre_image':
            if not analysis.pre_image_path:
                flash('Исходное изображение не найдено в базе данных', 'danger')
                return redirect(url_for('results', analysis_id=analysis_id))

            file_path = upload_dir / analysis.pre_image_path
            mime_type = 'image/jpeg'

        elif file_type == 'post_image':
            if not analysis.post_image_path:
                flash('Пост-катастрофическое изображение не найдено в базе данных', 'danger')
                return redirect(url_for('results', analysis_id=analysis_id))

            file_path = upload_dir / analysis.post_image_path
            mime_type = 'image/jpeg'

        else:
            flash('Неверный тип файла', 'warning')
            return redirect(url_for('results', analysis_id=analysis_id))

        # Проверяем существование файла
        if not file_path.exists():
            logger.error(f"Файл не найден на диске: {file_path}")
            flash('Файл не найден на сервере', 'danger')
            return redirect(url_for('results', analysis_id=analysis_id))

        # Отправляем файл
        return send_from_directory(
            directory=str(file_path.parent),
            path=file_path.name,
            as_attachment=True,
            mimetype=mime_type
        )

    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {str(e)}", exc_info=True)
        flash('Произошла ошибка при скачивании файла', 'danger')
        return redirect(url_for('results', analysis_id=analysis_id))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            current_user.full_name = request.form.get('full_name', current_user.full_name)
            current_user.organization = request.form.get('organization', current_user.organization)
            current_user.email = request.form.get('email', current_user.email)

            new_password = request.form.get('new_password')
            if new_password:
                if len(new_password) < 8:
                    flash('Пароль должен содержать минимум 8 символов', 'warning')
                else:
                    current_user.set_password(new_password)

            db.session.commit()
            flash('Профиль успешно обновлен', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Ошибка при обновлении профиля', 'danger')
            logger.error(f"Ошибка обновления профиля: {str(e)}", exc_info=True)

        return redirect(url_for('profile'))

    return render_template('profile.html')


@app.route('/api/results/<int:analysis_id>')
@login_required
def api_results(analysis_id):
    try:
        analysis = db.session.get(Analysis, analysis_id)
        if not analysis or analysis.user_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403

        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        result_json_path = upload_dir / analysis.result_json_path

        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        return jsonify(result_data)

    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to load results'}), 500


if __name__ == '__main__':
    app.run(debug=True)

