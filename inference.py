# %% h
import torch, cv2
import torchvision.transforms as transforms
from tqdm import tqdm
from model import Detr, feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])

model = Detr.load_from_checkpoint(
    "lightning_logs/m4r7ybn4/checkpoints/epoch=32-step=13068.ckpt"
)
model.to(device)
model.eval()
# %%

cap = cv2.VideoCapture("qc2mr.mp4")
frame_counter = 0
while cap.isOpened():
    frame_counter += 1
    ret, frame = cap.read()
    if frame_counter % 5 == 0:
        img_tensor = transform(frame).to(device).unsqueeze(0)
        outputs = model(img_tensor)
        probs = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probs.max(-1).values > 0.9
        target_sizes = torch.tensor(frame.shape[:-1]).unsqueeze(0)

        processed_outputs = feature_extractor.post_process(outputs, target_sizes)

        boxes = processed_outputs[0]["boxes"][keep]
        confidences = probs[keep]
        for confidence, (xmin, ymin, xmax, ymax) in zip(confidences, boxes.tolist()):
            print(confidence, (xmin, ymin, xmax, ymax))
            frame = cv2.rectangle(
                frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2
            )
        demo_frame = cv2.resize(frame, (864, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow("window-name", demo_frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cv2.destroyAllWindows()

# %%
