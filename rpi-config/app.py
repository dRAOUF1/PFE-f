import signal
from flask import Flask, jsonify, render_template,request
import subprocess
import requests
import socket, os
from dotenv import load_dotenv

app = Flask(__name__)
PID_FILE = 'script.pid'
#supprimer le fichier PID
if os.path.exists(PID_FILE):
    os.remove(PID_FILE)
@app.route('/config')
def index():
    load_dotenv(override=True) 
    default_values = {
        'mode': os.getenv('MODE', 'reseau'),
        'room_name': os.getenv('ROOM_NAME', ''),
        'app_url': os.getenv('URL_BACKEND', ''),
        'rotation': os.getenv('ROTATION', '0'),
        'detection_threshold': os.getenv('DETECTION_THRESHOLD', '0.65'),
        'recognition_threshold': os.getenv('RECOGNITION_THRESHOLD', '0.45'),
        'tracker_max_distance': os.getenv('TRACKER_MAX_DISTANCE', '90'),
        'tracker_max_frame_loss': os.getenv('TRACKER_MAX_FRAME_LOSS', '4')
    }
    # Rendre le fichier HTML index.html
    return render_template('config.html', default_values=default_values)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/connexion', methods=['POST'])
def connexion():
    # Récupérer les données du formulaire
    roomName = request.form['roomName']
    ssid = request.form['ssid']
    password = request.form['password']
    appURL = request.form['appURL']
    rotation = int(request.form['rotation'])
    detection_threshold = float(request.form['detection_threshold'])
    recognition_threshold = float(request.form['recognition_threshold'])
    tracker_max_distance = float(request.form['tracker_max_distance'])
    tracker_max_frame_loss = float(request.form['tracker_max_frame_loss'])
    mode = request.form["mode"]
    print(f"roomName: {roomName}, ssid: {ssid}, password: {password}, appURL: {appURL}, rotation: {rotation}, detection_threshold: {detection_threshold}, recognition_threshold: {recognition_threshold}, tracker_max_distance: {tracker_max_distance}, tracker_max_frame_loss: {tracker_max_frame_loss}")
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    

    
    
    try:
        with open('/home/pi/Desktop/reco/.env','w') as f:
            f.write(f"MODE={mode}\nROOM_NAME={roomName}\nURL_BACKEND={appURL}\nROTATION={rotation}\n DETECTION_THRESHOLD={detection_threshold/100}\nRECOGNITION_THRESHOLD={recognition_threshold}\nTRACKER_MAX_DISTANCE={tracker_max_distance}\nTRACKER_MAX_FRAME_LOSS={tracker_max_frame_loss}")
        with open('./.env','w') as f:
            f.write(f"MODE={mode}\nROOM_NAME={roomName}\nURL_BACKEND={appURL}\nROTATION={rotation}\n DETECTION_THRESHOLD={detection_threshold/100}\nRECOGNITION_THRESHOLD={recognition_threshold}\nTRACKER_MAX_DISTANCE={tracker_max_distance}\nTRACKER_MAX_FRAME_LOSS={tracker_max_frame_loss}")
    except FileNotFoundError:
        return jsonify({'message': "Fichier de configuration introuvable"})
        
        
    if mode=="reseau":
        #r = requests.post(f"{appURL}/addRaspberryPi",json={"salle":roomName,"addressIp":IPAddr})
        # Execute the command
        process = subprocess.Popen('nmcli con delete localnet', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the command to finish and get the return code
        return_code = process.wait()

        process = subprocess.Popen(f"sudo nmcli con add type wifi ifname 'wlan0' ssid '{ssid}' \
        con-name 'localnet' -- wifi-sec.key-mgmt 'wpa-psk' \
        wifi-sec.psk '{password}'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        process = subprocess.Popen('sleep 10 && sudo reboot', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return_code = process.wait()
        # if return_code != 0:
        #     return render_template('./erreur.html',erreur="Erreur lors de la création de la connexion au réseau wifi")
    
    return jsonify({'message': 'Coniguration effectué avec succès!'})
    
@app.route('/run-script', methods=['POST'])
def run_script():
    # Check if PID file exists
    if os.path.exists(PID_FILE):
        print("PID file exists")
        print("Killing old process")
        with open(PID_FILE, 'r') as f:
            old_pid = int(f.read().strip())
            print("old pid",old_pid)
            try:
                # Check if the process is still running
                os.kill(old_pid, 0)
                print("Process killed")
            except OSError:
                # Process is not running, ignore
                pass
            else:
                # Kill the old process
                os.kill(old_pid, signal.SIGTERM)
                print("Process killed")
    else:
        print("PID file does not exist")
    
    # Start new process with stdout and stderr captured
    # process = subprocess.Popen(
    #     ['sudo','pip', 'install','scipy', '--break-system-packages'],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE
    # )
    # stdout, stderr = process.communicate()
    # return_code = process.returncode

    process = subprocess.Popen(
        ['sudo','python', '/home/pi/Desktop/reco/main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    # print(process.pid)
    # Wait for the process to complete and capture stdout and stderr
    stdout, stderr = process.communicate()
    # print('message: Erreur lors du lancement du script!', 'error:', stderr.decode('utf-8'))
    return_code = process.returncode
   
    
    return jsonify({'message': 'Script lancé avec succès!', 'pid': process.pid})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4999,debug=True)
