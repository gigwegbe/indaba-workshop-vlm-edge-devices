from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image

# Load model and processor
model_id = "LiquidAI/LFM2-VL-450M"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# ===== LOAD TEST IMAGE =====
# image_path = "t8-ans.jpg"  # replace with your own image
image_path = "./detected/003843_detections.jpg" 
image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": """You are an expert automobile inspector. 
Review the image and verify whether the car damages shown are consistent with the following object detection results. 

Return the result strictly as JSON in the format below (no extra text, only JSON):

{   "short description of damages": "<>",
    "damage_types": "<e.g dent,scratch, crack, glass_shatter, lamp_broken, tire_flat>",
    "location_of_damages": "<>",
    "recommended_repair_work": "<>",
    "severity_of_damage": "<e.g., minor, medium, major>"
}"""
            },
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)


outputs = model.generate(**inputs, max_new_tokens=64 * 3)
decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the assistant's reply
if "assistant" in decoded:
    response = decoded.split("assistant", 1)[1].strip()
else:
    response = decoded.strip()

print("\n--- Model Response ---")
print(response)

