from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import io

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

openai.api_key = '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'

def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

def ask(A, num_captions=1):
    captions = []
    chat_history = [{"role": "system", "content": "Generate the caption of image for posting on social media on the basis of keywords given below:"}]
    
    for i in range(num_captions):
        chat_history.append({"role": "user", "content": A})
        chat_history_with_system = [{"role": "system", "content": "Generate the caption of image for posting on social media on the basis of keywords given below:"}] + chat_history
        completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=chat_history_with_system)
        generated_caption = completion['choices'][0]['message']['content']
        captions.append(f"{i+1}. {generated_caption}")
    
    return captions

@app.route('/generate-caption', methods=['POST'])
def generate_caption_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image.save("temp.png")

    try:
        keyword = generate_caption("temp.png")
        generated_captions = ask(keyword, num_captions=3)
        return jsonify({'keyword': keyword, 'captions': generated_captions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
