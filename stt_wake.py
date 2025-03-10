import os
import queue
import json
import threading
import pyaudio # 마이크 입력을 받기 위함
import wave
from google.cloud import speech
import numpy as np
import cv2 #  TV 화면 캡처를 위함
from datetime import datetime

# Google Cloud 설정 (환경 변수에 JSON 키 파일 경로 설정 필요)
current_directory = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(current_directory, "hyub_google_cloud_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# export GOOGLE_APPLICATION_CREDENTIALS="key/hyub_google_cloud_key.json"

# 오디오 스트림 설정
RATE = 16000 # 샘플링 주파수 (16kHz)
CHUNK = int(RATE / 10)  # 100ms 단위로 오디오 데이터를 처리

# Google Speech-to-Text 클라이언트
client = speech.SpeechClient()

# 오디오 입력 스트리밍을 위한 큐 (오디오 데이터를 실시간으로 저장할 버퍼(임시 저장 공간))
audio_queue = queue.Queue() 

# PyAudio 스트리밍 콜백 함수
def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data) # 입력된 오디오 데이터를 큐에 저장
    return (None, pyaudio.paContinue)

# PyAudio 스트리밍 설정
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

# Google STT 스트리밍 요청
def recognize_stream():
    """실시간 음성 인식을 수행하는 함수"""
    global trigger_detected
    global full_transcript
    
    print("🎤 STT 서비스 시작: '새미야', '쌤', '쌤아'를 감지 중...")
    
    trigger_words = ["세미야", "새미야", "쌤", "쌤아"] # 트리거 키워드 목록
    
    while True:
        # STT 요청에 필요한 오디오 설정
        requests = (speech.StreamingRecognizeRequest(audio_content=audio_queue.get()) for _ in range(10))
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="ko-KR"
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
        
        responses = client.streaming_recognize(streaming_config, requests)
        
        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript.strip()
                print(f"💬 인식된 문장: {transcript}")
                
                #  '새미야', '쌤', '쌤아' 감지
                if any(word in transcript for word in trigger_words):
                    print("🟢 호출 감지!")
                    if transcript == trigger_words:
                        print("🔊 응답: '네'")
                    else:
                        full_transcript = transcript
                        trigger_detected = True
                        return  # 음성 감지 후 종료

# TV 화면 캡쳐 함수 (OpenCV 사용)
def capture_tv_frame():
    """TV 화면을 24프레임(1초) 동안 캡쳐"""
    cap = cv2.VideoCapture(0)  # TV 화면을 캡처할 카메라 설정 (0번 웹캠 사용)
    frame_list = []

    print("📸 TV 화면 캡쳐 시작 (1초 동안 24프레임 저장)...")
    for _ in range(24):  
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
    
    cap.release()
    
    # 저장된 프레임 중 중앙 프레임 선택
    if frame_list:
        middle_frame = frame_list[len(frame_list) // 2]
        filename = f"tv_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, middle_frame)
        print(f"✅ 캡쳐 완료: {filename}")
        return filename

    return None

# 발화 내용 JSON 정형화
def parse_speech_to_json(speech_text, image_filename):
    """유저 발화 내용을 JSON 형태로 정리"""
    parsed_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "speech_text": speech_text,
        "captured_image": image_filename,
        "features": {
            "clothing_item": None,
            "color": None,
            "person": None
        }
    }

    # 특정 키워드 분석 (예시)
    keywords = {
        "color": ["검은색", "빨간색", "파란색", "흰색", "회색"],
        "clothing" : [
                        "반팔 셔츠",  # Short Sl. Shirt
                        "긴팔 셔츠",  # Long Sl. Shirt
                        "반팔 아우터",  # Short Sl. Outw.
                        "긴팔 아우터",  # Long Sl. Outw.
                        "조끼",  # Vest
                        "슬링탑",  # Sling
                        "반바지",  # Shorts
                        "긴바지",  # Trousers
                        "치마",  # Skirt
                        "반팔 원피스",  # Short Sl. Dress
                        "긴팔 원피스",  # Long Sl. Dress
                        "조끼 원피스",  # Vest Dress
                        "슬링 원피스"  # Sling Dress
                    ],

    }

    words = speech_text.split()
    for word in words:
        if word in keywords["color"]:
            parsed_data["features"]["color"] = word
        if word in keywords["clothing"]:
            parsed_data["features"]["clothing_item"] = word
        if "남자" in words:
            parsed_data["features"]["person"] = "남성"
        elif "여자" in words:
            parsed_data["features"]["person"] = "여성"

    return json.dumps(parsed_data, ensure_ascii=False, indent=4)

# 실행 함수
def main():
    global trigger_detected
    global full_transcript

    trigger_detected = False
    full_transcript = ""

    # STT 실행
    stt_thread = threading.Thread(target=recognize_stream)
    stt_thread.start()
    stt_thread.join()

    if trigger_detected:
        # TV 화면 캡처
        captured_image = capture_tv_frame()
        
        # 발화 내용 JSON 정형화
        structured_json = parse_speech_to_json(full_transcript, captured_image)
        print("\n📝 정리된 JSON 데이터:\n", structured_json)

if __name__ == "__main__":
    main()
