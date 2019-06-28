import utils, os, shutil, binascii
from flask import Flask, render_template, request, redirect, url_for, flash, \
    send_from_directory
from werkzeug.utils import secure_filename
from flask_session import Session
from utils import get_vox

app = Flask(__name__)
sess = Session()
app.secret_key = 'lmao'

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/data/<path:folder>/<path:filename>', methods=['GET'])
def download_file(folder, filename):
    folder_path = os.path.abspath('data/'+folder)
    file_path = folder_path + '/' + filename
    file_handle = open(file_path, 'r')

    return send_from_directory(directory=folder_path, filename=filename)
    # This *replaces* the `remove_file` + @after_this_request code above
    def stream_and_remove_file():
        yield from file_handle
        file_handle.close()
        shutil.rmtree(folder_path, ignore_errors=True)

    return app.response_class(
        stream_and_remove_file(),
        headers={'Content-Disposition': 'attachment', 'filename': filename}
    )

@app.route("/vocal", methods=["POST"])
def convert():
    try:
        file = request.files["file"]
        randgen = binascii.b2a_hex(os.urandom(4)).decode()
        filename = secure_filename(file.filename)
        folder = 'data/' + randgen
        os.makedirs(folder, exist_ok=True)
        loc = folder + '/' + filename
        file.save(loc)
        filename = filename.rsplit('.', 1)
        filename = folder + '/[VOXTRAC]' + filename[0] + '.wav'
        get_vox(loc, filename)
    except Exception as e:
        print(e)
        flash("Some error happened. Try again.\n")
        return redirect(url_for('home'))

    print(filename)
    return render_template("vocal.html",
            audio_orig='/'+loc,
            audio_parsed='/'+filename)

if __name__ == '__main__':
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    sess.init_app(app)
    app.run(host='0.0.0.0', port=5000, debug=True)