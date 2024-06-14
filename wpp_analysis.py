import re
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_whatsapp_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_data = file.read()
    return chat_data

def preprocess_chat(chat_data):

    chat_data = re.sub(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - .* criptografia .*', '', chat_data)
    
    pattern = r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}) - '
    messages = re.split(pattern, chat_data)

    if messages[0] == '':
        messages = messages[1:]
    
    dates = re.findall(pattern, chat_data)
    
    chat = []
    for date, message in zip(dates, messages):
        split_message = message.split(': ', 1)
        if len(split_message) == 2:
            author, text = split_message
        else:
            author = "System"
            text = split_message[0]
        chat.append([date.strip(), author.strip(), text.strip()])
    
    return chat

def analyze_sentiments(chat):
    for entry in chat:
        text = entry[2]
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        entry.append(sentiment)
    return chat

file_path = 'wpp-sentiment-analysis/conversa-wpp.txt'
chat_data = load_whatsapp_chat(file_path)

chat = preprocess_chat(chat_data)

print("Primeiras 5 mensagens extraídas:")
for line in chat[:5]:
    print(line)

chat_with_sentiments = analyze_sentiments(chat)

print("Primeiras 5 análises de sentimentos:")
for line in chat_with_sentiments[:5]:
    print(line)

df = pd.DataFrame(chat_with_sentiments, columns=['Date', 'Author', 'Message', 'Sentiment'])

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')

df = df.dropna(subset=['Date'])

df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

print("Verificação dos tipos de dados na coluna 'Sentiment':")
print(df['Sentiment'].dtype)
print(df['Sentiment'].head())

print("Estatísticas básicas de sentimentos:")
print(df['Sentiment'].describe())

df.set_index('Date', inplace=True)

fig, ax = plt.subplots(figsize=(17, 8))
ax.plot(df.index, df['Sentiment'])

ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

plt.title('Sentiment Analysis over Time')
plt.ylabel('Sentiment')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.grid(True)

plt.show()
