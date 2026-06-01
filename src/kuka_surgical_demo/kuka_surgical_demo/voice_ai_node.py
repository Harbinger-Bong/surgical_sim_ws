#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import os

class VoiceAINode(Node):
    def __init__(self):
        super().__init__('voice_ai_node')
        self.publisher_ = self.create_publisher(String, '/voice_command', 10)
        
        # Load Vosk model (ensure you have the 'model' directory in your path)
        # You can download 'vosk-model-small-en-us' and put it in your workspace
        model_path = os.path.join(os.path.dirname(__file__),"vosk-model-small-en-us")
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        
        self.get_logger().info('Vosk Voice AI initialized. Listening...')
        
        # Start audio stream
        self.stream = sd.RawInputStream(samplerate=16000, blocksize=8000, 
                                        dtype='int16', channels=1, callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        # 1. Capture the audio data
        data = bytes(indata)
        
        # 2. Check for Final result (Speech has paused)
        if self.rec.AcceptWaveform(data):
            result = json.loads(self.rec.Result())
            text = result.get('text', '')
            if text:
                self.get_logger().info(f"Final phrase detected: '{text}'")
                self.process_text(text)
        else:
            # 3. CATCH-ALL: Print Partial result (Speech in progress)
            partial = json.loads(self.rec.PartialResult())
            partial_text = partial.get('partial', '')
            if partial_text:
                # This will stream what the AI hears in real-time
                self.get_logger().info(f"Heard so far: '{partial_text}'", throttle_duration_sec=1.0)

    def process_text(self, text):
        # Catch-all: log everything that the AI considers "Final"
        self.get_logger().info(f"Processing final text: {text}")
        
        instruments = ['scalpel', 'forceps', 'retractor']
        found = False
        for instrument in instruments:
            if instrument in text:
                self.get_logger().info(f"MATCH FOUND: {instrument}")
                msg = String()
                msg.data = instrument
                self.publisher_.publish(msg)
                found = True
        
        if not found:
            self.get_logger().info("No instrument detected in phrase.")
            
def main(args=None):
    rclpy.init(args=args)
    node = VoiceAINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
