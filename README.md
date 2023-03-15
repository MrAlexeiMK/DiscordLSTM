# DiscordLSTM
 Train discort bot LSTM (Long short-term memory) model on your discord friends messages by using <a href="https://discord.com/developers/docs/topics/oauth2">Discord API</a>!
## Install
```pip install -r requirements.txt```

## Usage
### Collect messages dataset of user
```python bot_collect_messages.py -t <discordBotToken> -u [userId] -c [channelId] --limit [messagesLimit] -o [outputfile]```  

```discordBotToken``` - unique token of your discord bot  
```userId``` - discord id of user whose message you want to collect (enable <i>Developer mode</i> in discord settings, right click on user avatar and copy id)  
```channelId``` - id of text channel (right click - copy id on <i>Developer mode</i>)
```messagesLimit``` - how much user's messages you want to collect starting from the last  
```outputFile``` - file to store messages needed to train model  

### Train model on collected messages
```python bot_train.py --modelPath <modelPath> --dataPath <trainDataPath> -l [extractLimit] -e [epochsCount]```  

```modelPath``` - path to folder to save trained model  
```trainDataPath``` - path to '.csv' file with collected user messages  
```extractLimit``` - count of random rows from file that will been used to train model  
```epochsCount``` - count of epochs to train model

### Start bot
```python bot_start.py --token <discordBotToken> --model <modelPath>```  

```discordBotToken``` - unique token of your discord bot  
```modelPath``` - path to folder to save trained model  
