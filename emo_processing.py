import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.EmoSet2 import EmoSet2

if __name__ == '__main__':
    data_root = r"C:\Users\manee\Downloads\EmoSet-118K-7"
    num_emotion_classes = 8
    phase = 'val'
    print(data_root)

    dataset = EmoSet2(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase=phase,
    )

    info = dataset.get_info(data_root, num_emotion_classes)
    # print(dataset.info)
    dataloader = DataLoader(dataset, batch_size=1, shuffle = True)

    # METRICS
    count = 0
    hasEmotion = 0
    emotions = [0, 0, 0, 0, 0, 0, 0, 0]

    log_sep = 1000
    log_num = 0

    for i, data in enumerate(dataloader):
        # if i < 6:
        # Assuming data['image'] has the shape (1, 3, 224, 224)
        # print(data['image'])
        image = data['image'][0]  # Remove the batch dimension, shape becomes (3, 224, 224)
        image = image.permute(1, 2, 0)  # Transpose to shape (224, 224, 3)

        emotion = info['emotion']['idx2label'][str(data['emotion_label_idx'].paletz_item())]
        # print("Emotion: " + emotion)
        emotions[data['emotion_label_idx'].paletz_item()] += 1

        #Alt Attr processing
        """if str(data['scene_label_idx'].item()) != "-1":
            scene = info['scene']['idx2label'][str(data['scene_label_idx'].item())]
            # print("Scene: " + scene)

        if str(data['facial_expression_label_idx'].item()) != "-1":
            facial_expression = info['facial_expression']['idx2label'][str(data['facial_expression_label_idx'].item())]
            # print("Facial Expression: " + facial_expression)

        if str(data['human_action_label_idx'].item()) != "-1":
            human_action = info['human_action']['idx2label'][str(data['human_action_label_idx'].item())]
            # print("Human Action: " + human_action)

        if str(data['brightness_label_idx'].item()) != "-1":
            brightness = str(info['brightness']['idx2label'][str(data['brightness_label_idx'].item())])
            # print("Brightness: " + brightness)

        if str(data['colorfulness_label_idx'].item()) != "-1":
            colorfulness = str(info['colorfulness']['idx2label'][str(data['colorfulness_label_idx'].item())])
            # print("Colorfulness: " + colorfulness)

        objs_present = []
        for idx, val in enumerate(data['object_label_idx'][0].numpy()):
            if val == 1:
                object = info['object']['idx2label'][str(idx)]
                objs_present.append(object)
        if len(objs_present) > 0:
            pass
            # print("Objects Present:")
            # print(objs_present)
        """

        count += 1

        if i % log_sep == 0:
            log_num += 1
            print("Log Num: " + str(log_num))
            print("Count: " + str(count))
            print("# Amused: " + str(emotions[0]))

    print("Count: " + str(count))
    print("# Amusement: " + str(emotions[0]))
    print("# Awe: " + str(emotions[1]))
    print("# Contentment: " + str(emotions[2]))
    print("# Excitement: " + str(emotions[3]))
    print("# Anger: " + str(emotions[4]))
    print("# Disgust: " + str(emotions[5]))
    print("# Fear: " + str(emotions[6]))
    print("# Sadness: " + str(emotions[7]))