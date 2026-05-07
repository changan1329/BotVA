import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from Discriminator import DisLayer 
from Generator import AdversarialGraphGenerator 
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
import torch_geometric


class BotDataset(Dataset):
    def __init__(self, name, batch_size, user_num):
        path = "./datasets/Twibot-20/"
        self.name = name
        self.user_num = user_num

        self.label = torch.load(path + 'label.pt')
        follower_edge = torch.load(path + "follower_edge.pt")
        following_edge = torch.load(path + "following_edge.pt")
        self.edge_index_list = [follower_edge, following_edge]
        self.batch_size = batch_size

        self.user_features_numeric = torch.load(path + "num_properties_tensor.pt")
        self.user_features_bool = torch.load(path + "cat_properties_tensor.pt")
        self.user_features_tweet = torch.load(path + "tweets_tensor.pt")
        self.user_feature_des = torch.load(path + "des_tensor.pt")
        self.gen_features = torch.load(path + "generated_features.pt")
        self.gen_labels = torch.load(path + "generated_labels.pt")
        if self.name == "train":
            self.length = int(0.7 * user_num / self.batch_size)
        else:
            self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.user_features_numeric, self.user_features_bool, self.edge_index_list, \
               self.label, self.user_features_tweet, self.user_feature_des, self.gen_features, self.gen_labels


class ASRNBotDetector(pl.LightningModule):

    def __init__(self, HAN_hid_out, tweet_in_channel, numeric_in_channels, bool_in_channels,
                 des_in_channels, num_heads, semantic_heads, linear_out_channels, dropout_rate,
                 user_num, batch_size, adversarial_training=True, lambda_adv=1.0):
        super().__init__()

        self.batch_size = batch_size
        self.numeric_in_channels = numeric_in_channels
        self.bool_in_channels = bool_in_channels
        self.linear_out_channels = linear_out_channels
        self.tweet_in_channel = tweet_in_channel
        self.dropout_rate = dropout_rate
        self.user_num = user_num
        self.num_heads = num_heads
        self.des_in_channel = des_in_channels
        self.semantic_heads = semantic_heads
        self.adversarial_training = adversarial_training
        self.lambda_adv = lambda_adv
        
        if self.adversarial_training:
            self.automatic_optimization = False
            
        self.in_linear_numeric = nn.Linear(self.numeric_in_channels, int(self.linear_out_channels / 4), bias=True)
        self.in_linear_bool = nn.Linear(self.bool_in_channels, int(self.linear_out_channels / 4), bias=True)
        self.in_linear_tweet = nn.Linear(self.tweet_in_channel, int(self.linear_out_channels / 4), bias=True)
        self.in_linear_des = nn.Linear(self.des_in_channel, int(self.linear_out_channels / 4), bias=True)
        self.in_linear_gen = nn.Linear(1550, 128, bias=True)
        self.linear1 = nn.Linear(linear_out_channels, linear_out_channels)
        
        self.Dis_layers = nn.ModuleList()
        self.Dis_layers.append(DisLayer(num_edge_type=2, in_size=linear_out_channels, out_size=HAN_hid_out,
                                        layer_num_heads=num_heads[0], semantic_head=semantic_heads[0],
                                        dropout=dropout_rate))
        for l in range(1, len(num_heads)):
            self.Dis_layers.append(DisLayer(num_edge_type=2, in_size=HAN_hid_out, out_size=HAN_hid_out,
                                            layer_num_heads=num_heads[l], semantic_head=semantic_heads[l],
                                            dropout=dropout_rate))                                           
        self.output1 = nn.Linear(HAN_hid_out, 64)
        self.output2 = nn.Linear(64, 2)    
        if self.adversarial_training:
            self.generator = AdversarialGraphGenerator(
                node_features_dim=linear_out_channels,
                hidden_dim=128,
                dropout=dropout_rate
            )
        # Initialization
        self.apply(self._init_weights)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def encode_features(self, user_features_numeric, user_features_bool,
                        user_features_tweet, user_features_des, apply_dropout=True):
        if apply_dropout:
            f_num = self.dropout(self.ReLU(self.in_linear_numeric(user_features_numeric)))
            f_bool = self.dropout(self.ReLU(self.in_linear_bool(user_features_bool)))
            f_tweet = self.dropout(self.ReLU(self.in_linear_tweet(user_features_tweet)))
            f_des = self.dropout(self.ReLU(self.in_linear_des(user_features_des)))
        else:
            f_num = self.ReLU(self.in_linear_numeric(user_features_numeric))
            f_bool = self.ReLU(self.in_linear_bool(user_features_bool))
            f_tweet = self.ReLU(self.in_linear_tweet(user_features_tweet))
            f_des = self.ReLU(self.in_linear_des(user_features_des))
        user_features = torch.cat((f_num, f_bool, f_tweet, f_des), dim=1)
        if apply_dropout:
            user_features = self.dropout(self.ReLU(self.linear1(user_features)))
        else:
            user_features = self.ReLU(self.linear1(user_features))
        return user_features

    def discriminator_forward(self, user_features, edge_index_list, edge_weight_list=None, apply_dropout=True):
        for Dis in self.Dis_layers:
            if edge_weight_list is not None:
                user_features = Dis(user_features, edge_index_list, edge_weight_list)
            else:
                user_features = Dis(user_features, edge_index_list)
                
        if apply_dropout:
            user_features = self.dropout(self.ReLU(self.output1(user_features)))
        else:
            user_features = self.ReLU(self.output1(user_features))
        pred = self.output2(user_features)
        return pred

    def training_step(self, train_batch, batch_idx):
        user_features_numeric = train_batch[0].squeeze(0)
        user_features_bool = train_batch[1].squeeze(0)
        edge_index_list = train_batch[2]
        label = train_batch[3].squeeze(0).tolist()
        user_features_tweet = train_batch[4].squeeze(0).squeeze(1)
        user_features_des = train_batch[5].squeeze(0).squeeze(1)
        gen_features = train_batch[6].squeeze(0).squeeze(1)
        gen_labels = train_batch[7].squeeze(0).tolist()
        label = torch.LongTensor(label[0:int(0.7 * self.user_num)])
        gen_labels = torch.LongTensor(gen_labels)
        label = torch.cat((label, gen_labels), dim=0)
    
        batch_id = torch.randperm(int(0.7 * self.user_num))[0:self.batch_size].tolist()
        batch_labels = label[batch_id].cuda()
    
        user_features = self.encode_features(user_features_numeric, user_features_bool,
                                             user_features_tweet, user_features_des, apply_dropout=True)
        gen_features = self.dropout(self.ReLU(self.in_linear_gen(gen_features)))
        user_features = torch.cat((user_features, gen_features), dim=0)
        if not self.adversarial_training:
            pred = self.discriminator_forward(user_features, edge_index_list, apply_dropout=True)
            pred = pred[batch_id]
            loss = self.CELoss(pred, batch_labels)
            accuracy = get_metrics(pred, batch_labels)
            F1 = F1_score(pred, batch_labels)
            self.log("train_loss", loss)
            self.log("train_accuracy", accuracy)
            self.log("train_F1", F1)
            return loss
        else:
            discriminator_opt, generator_opt = self.optimizers()
            if batch_idx % 2 == 0:
                discriminator_opt.zero_grad()
                with torch.no_grad():
                    adv_indices, adv_weights = self.generator(
                        user_features, edge_index_list, self.user_num
                    )
                pred_adv = self.discriminator_forward(user_features, adv_indices, adv_weights, apply_dropout=True)
                pred_adv = pred_adv[batch_id]
                pred_orig = self.discriminator_forward(user_features, edge_index_list, None, apply_dropout=True)
                pred_orig = pred_orig[batch_id]
                loss_adv = self.CELoss(pred_adv, batch_labels)
                loss_orig = self.CELoss(pred_orig, batch_labels)
                discriminator_loss = loss_orig + self.lambda_adv * loss_adv
                self.manual_backward(discriminator_loss)
                discriminator_opt.step()
                acc_orig = get_metrics(pred_orig, batch_labels)
                acc_adv = get_metrics(pred_adv, batch_labels)
                self.log("D_loss", discriminator_loss)
                self.log("D_acc_adv", acc_adv)
            else:
                generator_opt.zero_grad()
                adv_indices, adv_weights = self.generator(
                    user_features.detach(), edge_index_list, self.user_num
                )
                pred_adv = self.discriminator_forward(user_features.detach(), adv_indices, adv_weights, apply_dropout=False)
                pred_adv = pred_adv[batch_id]
                generator_loss = -self.CELoss(pred_adv, batch_labels)
                self.manual_backward(generator_loss)
                generator_opt.step()
                
                self.log("G_loss", generator_loss)

    def validation_step(self, val_batch, batch_idx):
        user_features_numeric = val_batch[0].squeeze(0)
        user_features_bool = val_batch[1].squeeze(0)
        edge_index_list = val_batch[2]
        label = val_batch[3].squeeze(0).tolist()
        user_features_tweet = val_batch[4].squeeze(0).squeeze(1)
        user_features_des = val_batch[5].squeeze(0).squeeze(1)
        label = torch.LongTensor(label[int(0.7 * self.user_num): int(0.9 * self.user_num)])
        
        user_features = self.encode_features(user_features_numeric, user_features_bool,
                                             user_features_tweet, user_features_des, apply_dropout=False)
        pred = self.discriminator_forward(user_features, edge_index_list, None, apply_dropout=False)
        pred = pred[int(0.7 * self.user_num): int(0.9 * self.user_num)]

        loss = self.CELoss(pred, label.cuda())
        accuracy = get_metrics(pred, label.cuda())
        F1 = F1_score(pred, label.cuda())
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)
        self.log("val_F1", F1)
        return accuracy

    def configure_optimizers(self):
        if not self.adversarial_training:
            optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WD)
            return optimizer
        else:
            discriminator_params = []
            generator_params = []
            for name, param in self.named_parameters():
                if 'generator' in name:
                    generator_params.append(param)
                else:
                    discriminator_params.append(param)
            
            d_opt = torch.optim.AdamW(discriminator_params, lr=LEARNING_RATE, weight_decay=WD)
            g_opt = torch.optim.AdamW(generator_params, lr=LEARNING_RATE, weight_decay=WD)
            return [d_opt, g_opt]

    def test_step(self, test_batch, batch_idx):
        user_features_numeric = test_batch[0].squeeze(0)
        user_features_bool = test_batch[1].squeeze(0)
        edge_index_list = test_batch[2]
        label = test_batch[3].squeeze(0).tolist()
        user_features_tweet = test_batch[4].squeeze(0).squeeze(1)
        user_features_des = test_batch[5].squeeze(0).squeeze(1)
        label = torch.LongTensor(label[int(0.9 * self.user_num):])
        
        user_features = self.encode_features(user_features_numeric, user_features_bool,
                                             user_features_tweet, user_features_des, apply_dropout=False)
        pred = self.discriminator_forward(user_features, edge_index_list, None, apply_dropout=False)
        pred = pred[int(0.9 * self.user_num):]
        
        loss = self.CELoss(pred, label.cuda())
        accuracy = get_metrics(pred, label.cuda())
        F1 = F1_score(pred, label.cuda())
        
        self.log("test_accuracy", accuracy)
        self.log("test_F1", F1)
        return accuracy

LEARNING_RATE = 1e-3
WD = 3e-5
DROPOUT_RATE = 0.5
USER_NUM = 11826
BATCH_SIZE = 256
IN_FEATURES_BOOL = 7
IN_FEATURES_NUMERIC = 7
IN_FEATURES_TWEET = 768
IN_FEATURE_DES = 768
MAX_EPOCHS = 40
NUM_HEAD = [8, 8]
SEMANTIC_HEAD = [8, 8]
ADVERSARIAL_TRAINING = True
LAMBDA_ADV = 1.0

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    train_dataset = BotDataset(name="train", batch_size=BATCH_SIZE, user_num=USER_NUM)
    valid_dataset = BotDataset(name="dev", batch_size=1, user_num=USER_NUM)
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(valid_dataset, batch_size=1)
    model = ASRNBotDetector(HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET,
                            numeric_in_channels=IN_FEATURES_NUMERIC, bool_in_channels=IN_FEATURES_BOOL,
                            linear_out_channels=128, dropout_rate=DROPOUT_RATE, user_num=USER_NUM,
                            batch_size=BATCH_SIZE, num_heads=NUM_HEAD, des_in_channels=IN_FEATURE_DES,
                            semantic_heads=SEMANTIC_HEAD, adversarial_training=ADVERSARIAL_TRAINING,
                            lambda_adv=LAMBDA_ADV)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=3,
        verbose=True
    )
    trainer = pl.Trainer(devices=1, num_nodes=1, max_epochs=MAX_EPOCHS, precision=32,
                         callbacks=[checkpoint_callback], log_every_n_steps=1)
    print("Training ASRN model begin!")
    trainer.fit(model, train_loader, val_loader)
    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path1 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_path2 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[1])
    best_path3 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[2])
    print('best_path1:', best_path1)
    print('best_path2', best_path2)
    print('best_path3', best_path3)
    best_model1 = ASRNBotDetector.load_from_checkpoint(checkpoint_path=best_path1,
                                                       HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET,
                                                       numeric_in_channels=IN_FEATURES_NUMERIC,
                                                       bool_in_channels=IN_FEATURES_BOOL,
                                                       linear_out_channels=128, dropout_rate=DROPOUT_RATE,
                                                       user_num=USER_NUM,
                                                       batch_size=BATCH_SIZE, num_heads=NUM_HEAD,
                                                       des_in_channels=IN_FEATURE_DES,
                                                       semantic_heads=SEMANTIC_HEAD,
                                                       adversarial_training=ADVERSARIAL_TRAINING,
                                                       lambda_adv=LAMBDA_ADV)
    best_model2 = ASRNBotDetector.load_from_checkpoint(checkpoint_path=best_path2,
                                                       HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET,
                                                       numeric_in_channels=IN_FEATURES_NUMERIC,
                                                       bool_in_channels=IN_FEATURES_BOOL,
                                                       linear_out_channels=128, dropout_rate=DROPOUT_RATE,
                                                       user_num=USER_NUM,
                                                       batch_size=BATCH_SIZE, num_heads=NUM_HEAD,
                                                       des_in_channels=IN_FEATURE_DES,
                                                       semantic_heads=SEMANTIC_HEAD,
                                                       adversarial_training=ADVERSARIAL_TRAINING,
                                                       lambda_adv=LAMBDA_ADV)
    best_model3 = ASRNBotDetector.load_from_checkpoint(checkpoint_path=best_path3,
                                                       HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET,
                                                       numeric_in_channels=IN_FEATURES_NUMERIC,
                                                       bool_in_channels=IN_FEATURES_BOOL,
                                                       linear_out_channels=128, dropout_rate=DROPOUT_RATE,
                                                       user_num=USER_NUM,
                                                       batch_size=BATCH_SIZE, num_heads=NUM_HEAD,
                                                       des_in_channels=IN_FEATURE_DES,
                                                       semantic_heads=SEMANTIC_HEAD,
                                                       adversarial_training=ADVERSARIAL_TRAINING,
                                                       lambda_adv=LAMBDA_ADV)
    print('Testing best_model1 (Adversarial trained discriminator)')
    trainer.test(best_model1, dataloaders=val_loader, verbose=True)
    print('Testing best_model2 (Adversarial trained discriminator)')
    trainer.test(best_model2, dataloaders=val_loader, verbose=True)
    print('Testing best_model3 (Adversarial trained discriminator)')
    trainer.test(best_model3, dataloaders=val_loader, verbose=True)
