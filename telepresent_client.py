# use a camera feed and send to webRTC
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time
import fractions
import sys
import traceback

# https://python-sounddevice.readthedocs.io/en/0.3.12/
import sounddevice

import numpy as np

import cv2
from aiohttp import web, ClientSession, ClientWebSocketResponse, WSMsgType

from av import AudioFrame, VideoFrame
from av.frame import Frame

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

from aiortc import RTCPeerConnection, RTCSessionDescription, \
    VideoStreamTrack, MediaStreamTrack, RTCConfiguration, RTCIceServer # , MediaStreamError
from aiortc.contrib.media import MediaPlayer, \
    MediaRecorder, PlayerStreamTrack
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp


ROOT = os.path.dirname(__file__)
CONFIG = None

logger = logging.getLogger("telepresent")
pcs = set()
PC = None

USER_NAME = 'ROBOT_python'


class CustomAudioStreamTrack(MediaStreamTrack):
    """
    A audio track which reads input.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._start = 5 #time.time()
        # in, out ?
        #sounddevice.default.samplerate = (16000, 48000)
        #sounddevice.default.dtype = ('int16', 'int32')
        #sounddevice.default.channels = (1, 1)
        # 7 C922 Pro Stream Webcam: USB Audio (hw:2,0), ALSA (2 in, 0 out)
        # NOTE: this should change to whatever mic will be used... respeaker?
        devs = sounddevice.query_devices()
        logger.info(devs)
        # sounddevice.default.device = 7
        #devs = sounddevice.query_devices()
        #logger.info(devs)
        self.stream = sounddevice.InputStream(device=sounddevice.default.device,
                                              samplerate=16000,
                                              dtype='int16',
                                              channels=1)
        self.stream.start()

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.
        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise Exception("media stream error") # MediaStreamError

        sample_rate = 16000 # 8000
        samples = int(AUDIO_PTIME * sample_rate)
        # logger.info("WTF!!")
        if hasattr(self, "_timestamp"):
            self._timestamp += samples
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            # logger.info("wating... {0}".format(wait))
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        # await asyncio.sleep(0.01)
        try:
            
            # data = sounddevice.rec(frames=samples, samplerate=sample_rate, channels=1, dtype='int16', blocking=True)
            data, overflowed = self.stream.read(frames=samples)
            #print("data", data)
            #print("data", data.shape)
            #data = np.zeros((2,1152), dtype='int16')
            #logger.info(data)
            
            # frameb = AudioFrame(format="s16", layout="mono", samples=samples)
            # logger.info('*****************************')
            # alt_data = frameb.to_ndarray()
            # logger.info(alt_data)
            # logger.info(alt_data.shape)
            # logger.info('-----------------')
            fixed_data = np.swapaxes(data, 0, 1)
            #print(fixed_data.shape)
            #print(fixed_data.dtype)
            #print(data.dtype)
            #print(fixed_data.dtype)
            frame = AudioFrame.from_ndarray(fixed_data, layout="mono", format='s16') # layout="stereo", format='s32') # s16
            #for p in frame.planes:
            #    p.update(data.tobytes()) #bytes(p.buffer_size))
            # logger.info(frame.planes)
            # logger.info("\n!!!!!!!!!!!")
            frame.pts = self._timestamp
            frame.sample_rate = sample_rate
            frame.time_base = fractions.Fraction(1, sample_rate)
        except Exception as exc:
            print(exc)
            logger.exception(exc)
        return frame
    
    def close(self):
        self.stream.stop()

class OpenCVVideoStreamTrack(MediaStreamTrack):
    """
    A video stream track.
    """

    kind = "video"

    def __init__(self, video_device, options):
        super().__init__()
        self._timestamp = 0
        self._start = time.time()
        self.video_device = video_device
        self.options = options
        # opencv cam capture
        self.cam = cv2.VideoCapture(self.video_device or 0)

    async def next_timestamp(self):
        if self.readyState != "live":
            raise Exception('MediaStreamError') #MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE
    
    def close(self):
        if self.cam:
            self.cam.release()

    async def recv(self):
        """
        Receive the next :class:`~av.video.frame.VideoFrame`.
        The base implementation just reads a 640x480 green frame at 30fps,
        subclass :class:`OpenCVVideoStreamTrack` to provide a useful implementation.
        """
        pts, time_base = await self.next_timestamp()
        
        while True:
            ret_val, img = self.cam.read()
            if ret_val:
                frame = VideoFrame.from_ndarray(img, format="bgr24")
                        #VideoFrame(width=self.options['width'] or 640,
                        #           height=self.options['height'] or 480)
                #for p in frame.planes:
                #    p.update(bytes(p.buffer_size))
                frame.pts = pts
                frame.time_base = time_base
                return frame


async def setup_peer():
    print("creating peer")
    global PC, GLOBAL

    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:coturn.retrowave.tech"]),
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=['turn:coturn.retrowave.tech'],
                     credential='password1',
                     username='username1')

    ])
    PC = RTCPeerConnection(config)
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(PC)


    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    # log_info("Created for %s", request.remote)


    @PC.on("datachannel")
    async def on_datachannel(channel):

        @channel.on("message")
        def on_message(message):
            # print(channel.label)
            if channel.label == 'gamepad':
                pass
                # print("GAMEPAD MESSAGE", message)
                # Example Message
                # {"id":"046d-c21f-Logitech Gamepad F710","state":{"FACE_1":0,"FACE_2":0,"FACE_3":0,"FACE_4":0,
                #  "LEFT_TOP_SHOULDER":0,"RIGHT_TOP_SHOULDER":0,"LEFT_BOTTOM_SHOULDER":0,"RIGHT_BOTTOM_SHOULDER":1,
                #  "SELECT_BACK":0,"START_FORWARD":0,"LEFT_STICK":0,"RIGHT_STICK":0,"DPAD_UP":0,"DPAD_DOWN":0,
                #  "DPAD_LEFT":0,"DPAD_RIGHT":0,"HOME":0,"LEFT_STICK_X":0,"LEFT_STICK_Y":0,"RIGHT_STICK_X":0,"RIGHT_STICK_Y":0},"ts":87090}
                #
                # msg_data = json.loads(message)
                # gp_state = msg_data['state']
                # send over LCM
                # msg = twist_t()
                # msg.timestamp = msg_data['ts']
                # msg.linear = (gp_state['LEFT_STICK_X'], gp_state['LEFT_STICK_Y'], 0.0)
                # msg.angular = (gp_state['RIGHT_STICK_X'], gp_state['RIGHT_STICK_Y'], 0.0)
                # msg.name = msg_data['id']
                
            else:
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])


    @PC.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", PC.iceConnectionState)
        if PC.iceConnectionState == "failed":
            await PC.close()
            pcs.discard(PC)
    
    @PC.on("icegatheringstatechange")
    async def icegatheringstatechange():
        print("icegatherstatechange")
        print(PC.iceGatheringState)
        await asyncio.sleep(0.1)

    print("peer created")
    print(PC)
    return PC


async def accept_offer(_sdp, _type):
    offer = RTCSessionDescription(sdp=_sdp, type=_type)

    options = {"framerate": "10", "video_size": "640x480"}
    # player = MediaPlayer("/dev/video0", format="v4l2", options=options)
    audio_track = CustomAudioStreamTrack()
    #"/dev/video0"
    video_track = OpenCVVideoStreamTrack(video_device=0, options=options)

    # handle offer
    await PC.setRemoteDescription(offer)
    # await recorder.start()

    # dont wait to get track, we only want to send out
    for t in PC.getTransceivers():
        if t.kind == "audio" and audio_track:
            PC.addTrack(audio_track)
        # elif t.kind == "video" and player.video:
        elif t.kind == "video" and video_track:
            PC.addTrack(video_track)

    # send answer
    answer = await PC.createAnswer()
    await PC.setLocalDescription(answer)

    return answer


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def handle_answer(msg_data):
    print("handling answer")
    print(msg_data)
    await asyncio.sleep(0.1)
    return "who knows"
          


## signaling server connection
async def websocket_session(session):
    logger.info("ws conection start")
    partner = None
    global PC, CONFIG
    try:
        async with session.ws_connect('ws://127.0.0.1:5000/ss') as ws:
            print("ws connected, sending join...")
            await ws.send_str(json.dumps({'type':'login', 'name': USER_NAME}))
            print("join sent listening to responses")
            
            while True:
                print('alive')
                async for msg in ws:
                    #print(msg)
                    if msg.type == WSMsgType.TEXT:
                        msg_data = json.loads(msg.data)
                        print("parsing messages")
                        if 'type' in msg_data:
                            try:
                                typ = msg_data['type']
                                if typ == 'login':
                                    if msg_data['success']:
                                        print('login success, creating peer connection')
                                        await setup_peer()
                                        print("now what...")
                                elif typ == 'offer':
                                    print("got an offer from: {0}".format(msg_data['name']))
                                    offer = msg_data['offer']
                                    partner = msg_data['name']
                                    answer = await accept_offer(offer['sdp'], offer['type'])
                                    await ws.send_str(json.dumps({'type': 'answer', 'name':msg_data['name'], 'answer': {"sdp": answer.sdp, "type": answer.type}}))
                                    print("answer sent to offer")
                                elif typ == 'answer':
                                    answer = await handle_answer(msg_data)
                                    # await ws.send_str(json.dumps({'type': 'answer', 'answer': answer}))
                                elif typ == 'candidate':
                                    print("got candidate: ", msg_data['candidate'])
                                    m_can = msg_data["candidate"]
                                    if len(m_can['candidate']) > 1:
                                        candidate = candidate_from_sdp(m_can['candidate'].split(":", 1)[1])
                                        candidate.sdpMid = m_can["sdpMid"]
                                        candidate.sdpMLineIndex = m_can["sdpMLineIndex"]
                                        print(candidate)
                                        PC.addIceCandidate(candidate)
                                elif typ == 'info':
                                    print("got info: {0}".format(msg_data))
                                    await asyncio.sleep(0.1)
                                elif typ == 'leave':
                                    if msg_data['name'] == partner:
                                        print("my partner {0} left".format(partner))
                                        partner = None
                                        await PC.close()
                                        # reset for next time
                                        await setup_peer()
                                else:
                                    print('not sure what to do...')
                                    print(msg_data)
                                    await asyncio.sleep(0.1)
                            except Exception as exc:
                                print(exc)
                                traceback.print_exc(file=sys.stdout)
                                await asyncio.sleep(0.5)

                    if msg.type in (WSMsgType.CLOSED,
                                    WSMsgType.ERROR):
                        print("closed or errro")
                        break
                print("no message")
                await asyncio.sleep(0.1)
    except Exception as exc:
        raise exc



async def init(app):
    session = ClientSession()
    #await websocket_session(session)
    app['websocket_task'] = app.loop.create_task(websocket_session(session))


def main():
    global CONFIG
    CONFIG = load_config()

    app = web.Application()
    app.on_startup.append(init)
    app.on_shutdown.append(on_shutdown)
   
    #loop = asyncio.get_event_loop()
    #future = asyncio.run_coroutine_threadsafe(do_handle(), loop)

    loop = asyncio.get_event_loop()
    try:
        # asyncio.ensure_future(do_handle())
        # asyncio.ensure_future(pub_buff())
        tasks = [web.run_app(app)]
        loop.run_until_complete(asyncio.wait(tasks))
        # loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()

def load_config(filename="config.json"):
    cfp = os.path.join(os.path.abspath(os.getcwd()), filename)
    try:
        return json.load(open(cfp))
    except Exception as exc:
        print(exc)
        return None

if __name__ == "__main__":
    main()
