from utils.getter import *


def train(model, dataloaders_dict, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
            traced.save('model.pth')
            best_acc = epoch_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--episode_dir' , default='../lux-episodes', type=str, help='project file that contains parameters')
    parser.add_argument('--split_ratio', type=float, default=0.1,
                    help='ratio of the test set (default: 0.1)')
    parser.add_argument('--seed_value', type=int, default=2021,
                    help='random seed value (default: 2021)')

    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for training (exponential of 2)')
    parser.add_argument('--loss_name' , default='ce', type=str, help='[ce | smoothce | focal]')
    parser.add_argument('--opt_lr' , default=1e-3, type=float, help='optimizer learning rate')
    parser.add_argument('--n_epochs' , default=20, type=int, help='number of epochs for training')


    args = parser.parse_args()

    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    seed_everything(args.seed_value)

    dataloaders_dict = get_dataloader_from_json(args)

    model = LuxNet()
    criterion = get_loss(args.loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train(model, dataloaders_dict, criterion, optimizer, args.n_epochs)