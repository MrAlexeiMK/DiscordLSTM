# DiscordLSTM
 Train discort bot by different models on your discord friends messages by using <a href="https://discord.com/developers/docs/topics/oauth2">Discord API</a>!
## Install
```pip install -r requirements.txt```

## Usage
### Collect messages dataset of user
```
python bot_collect_messages.py -t <discordBotToken> -u [userId] -c [channelId] --limit [messagesLimit] -o [outputfile]
```  

```discordBotToken``` - unique token of your discord bot  
```userId``` - discord id of user whose message you want to collect (enable <i>Developer mode</i> in discord settings, right click on user avatar and copy id). If not specified, all users will be selected  
```channelId``` - id of text channel (right click - copy id on <i>Developer mode</i>). If not specified, all text channels will be selected  
```messagesLimit``` - how much user's messages you want to collect starting from the last (100000 by default)  
```outputFile``` - file to store messages needed to train model (<i>data/discord_conversation_{userId}.csv</i> by default)  

### Train model on collected messages
```
python bot_train.py --modelPath <modelPath> --dataPath <trainDataPath> -l [extractLimit] -e [epochsCount] --splits [validationSplitsCount]
```  

```modelPath``` - path to folder to save trained model  
```trainDataPath``` - path to '.csv' file with collected user messages  
```extractLimit``` - count of random rows from file that will been used to train model (200000 by default)  
```epochsCount``` - count of epochs to train the model (50 by default)
```validationSplitsCount``` - count of train-test K-fold validation splits for each epoch (5 by default)  

Final count of training iterations will be <b>epochsCount * validationSplitsCount</b>.  
  
- Model is automatically saved every 10 epochs
- You can press <i>Ctrl + C</i> and it also will be saved
- Validation split needed to make sure the model just doesn't remember the test dataset
- After stopping you can run training again on the same dataset
- You can't retrain trained model on different dataset (only if this dataset contains same set of words)

### Start bot
```
python bot_start.py --token <discordBotToken> --model <modelPath
```  

```discordBotToken``` - unique token of your discord bot  
```modelPath``` - path to folder to save trained model  

Type <b>```/lstm [message]```</b> in your discord text channel and wait for bot answer!
