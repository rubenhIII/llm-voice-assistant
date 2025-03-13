import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torch.nn.functional import normalize

from kokoro import KPipeline
import pyttsx3

from llama_cpp import Llama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

class AIassistant:
    def __init__(self):
        self.record_sample_rate = 16000
        self.attention_message = "Sí, dime"
        self.bye_message = "Nos vemos!"
        self.assistant_name = "alana"
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant with name {assistant_name} that helps with brief and simple answers avoiding bullets in enumerations.",
            ),
            ("human", "{transcription}"),
        ])

        load_dotenv()
        # Load Models and Processors
        # Processor Audio to Text
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="spanish", task="transcribe")
        # Load Llama model
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Text to Audio
        self.pipeline = KPipeline(lang_code='e')
        self.engine_t2a = pyttsx3.init()

    def record(self, duration=5):
        # Recording duration
        recording = sd.rec(int(duration * self.record_sample_rate), 
                           samplerate=self.record_sample_rate, channels=1)
        print(f"Recording for {duration} seconds")
        sd.wait()
        print("... finished")
        
        recording = recording[:,0]
        #samplerate, recording = wavfile.read("Hola.wav")
        speech = torch.FloatTensor(recording)
        return speech
      
    def get_text_from_audio(self, speech):
        speech = speech.unsqueeze(0)
        speech = normalize(speech)
        speech = speech.squeeze(0)
    
        inputs = self.processor(speech, sampling_rate=self.record_sample_rate, 
                                return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features
        # generate token ids
        predicted_ids = self.model.generate(input_features, 
                                            forced_decoder_ids=self.forced_decoder_ids, 
                                            attention_mask=inputs["attention_mask"])
        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription
    
    def get_audio_from_text(self, text, offline=False):
        if not offline:
            generator = self.pipeline(
            text, voice='af_heart', # <= change voice here
            speed=1, split_pattern=r'\n+'
            )
        
            for i, (gs, ps, audio) in enumerate(generator):
                audio_np = audio.numpy()
                sd.play(audio_np, 24000)
                sd.wait()
                #print(i)  # i => index
                #print(gs) # gs => graphemes/text
                #print(ps) # ps => phonemes
                #display(Audio(data=audio, rate=24000, autoplay=i==0))
                #sf.write(f'{i}.wav', audio, 24000) # save each audio file

        self.engine_t2a.say(text)
        self.engine_t2a.runAndWait
        
    def run_pipeline(self):
        speech = self.record()
        transcription = self.get_text_from_audio(speech=speech)       
        chain = self.prompt | self.llm
        ai_msg = chain.invoke({"transcription": transcription, "assistant_name": self.assistant_name})
        ai_msg = ai_msg.content
        
        self.get_audio_from_text(text=ai_msg)

    def keep_aware(self):
        speech = self.record(duration=2)
        transcription = self.get_text_from_audio(speech=speech)
        print(transcription)
        if self.assistant_name in transcription.lower():
            self.get_audio_from_text(self.attention_message)
            self.run_pipeline()

