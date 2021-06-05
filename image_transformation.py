from torchvision import transforms

# data augmentation transformation
aug_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
# standard data transformation
standard_transform = transforms.Compose(
        [
            transforms.Resize((int(224 * 1.143), int(224 * 1.143))),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )