from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from predict import predict_single_data_category, predict_single_data_status
from constant import *

app = Flask(__name__)
CORS(app)

# Kategori tahminleme endpoint'i
@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        # POST isteğinden resim dosyasını al
        file = request.files['image']

        # Dosyanın geçerli bir dosya adına sahip olup olmadığını kontrol et
        if file and allowed_file_type(file.filename):
            # Resmi geçici bir dosyaya kaydet
            image_path = f"../data/{file.filename}"
            file.save(image_path)
           
            # Modeli yükleme ve kategori tahmini yapma
            model_path = MOBILE_NET_RMSprop

            result = predict_single_data_category(model_path, image_path, IMG_SIZE, CATEGORIES)

            # Geçici dosyayı silme
            os.remove(image_path)

            # Sonucu JSON formatında döndürme
            return jsonify(type=result)

        else:
            return jsonify(error="Geçersiz dosya formatı. Desteklenen formatlar: ['jpg', 'jpeg', 'png']")

    except Exception as e:
        return jsonify(error=str(e))

# Durum tahminleme endpoint'i
@app.route('/predict_status', methods=['POST'])
def predict_status():
    try:
        # POST isteğinden resim dosyasını al
        file = request.files['image']

        # Dosyanın geçerli bir dosya adına sahip olup olmadığını kontrol et
        if file and allowed_file_type(file.filename):
            # Resmi geçici bir dosyaya kaydet
            image_path = f"../data/{file.filename}"
            file.save(image_path)
           
            # Modeli yükleme ve durum tahmini yapma
            model_path = MOBILE_NET_RMSprop

            result = predict_single_data_status(model_path, image_path, IMG_SIZE)

            # Geçici dosyayı silme
            os.remove(image_path)

            # Sonucu JSON formatında döndürme
            return jsonify(result=result)

        else:
            return jsonify(error="Geçersiz dosya formatı. Desteklenen formatlar: ['jpg', 'jpeg', 'png']")

    except Exception as e:
        return jsonify(error=str(e))

# Dosya türünün izin verilen bir tür olup olmadığını kontrol etme fonksiyonu
def allowed_file_type(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Uygulamayı hata ayıklama modunda çalıştırma
    app.run(debug=True)
