from flask import Flask, request, Response
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS

from main import generate_pose

app = Flask(__name__)
app.secret_key = "Slh49OCql6yG"
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/pose", methods=['GET', 'POST'])
def handle_pose_estimation(payload):
    print(payload)
    print(request.__dict__)
    return Response(
        generate_pose(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/startRealTimeStream", methods=['GET', 'POST'])
def handle_start_real_time_stream():
    return Response(
        generate_pose(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/startStream/<pose_name>", methods=['GET', 'POST'])
def handle_start_stream(pose_name):
    print(pose_name)
    print(request.__dict__)
    return Response(
        generate_pose(pose_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@socketio.on("connect")
def handle_connect():
    print(request.sid)
    print("Client has connected")
    emit("connect", {"data": f"id: {request.sid} is connected"})


@socketio.on("data")
def handle_transfer_data(data):
    print(f"Data from the front-end {str(data)}")
    emit("data", {'data': data, 'id': request.sid}, broadcast=True)


@socketio.on("disconnect")
def handle_transfer_data():
    print(f"User disconnected")
    emit("disconnect", f"User {request.sid} disconnected", broadcast=True)


@socketio.on("message")
def handle_message(message):
    print(message)
    send(message, broadcast=True)
    return None


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
