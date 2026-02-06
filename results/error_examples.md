# Error Taxonomy Analysis

## Summary Statistics

| dataset   |   ambiguous_business |   char_substitution |   currency_symbols |   false_negative_other |   false_positive_other |   long_subtle_spam |   short_text |   template_marketing |   url_obfuscation |
|:----------|---------------------:|--------------------:|-------------------:|-----------------------:|-----------------------:|-------------------:|-------------:|---------------------:|------------------:|
| sms       |                    0 |                   0 |                  0 |                      1 |                      0 |                  0 |            1 |                    0 |                 1 |
| telegram  |                    1 |                   6 |                  1 |                     10 |                     18 |                  6 |            5 |                    3 |                 0 |

## Error Type Distribution

### SMS

| error_type   |   false_negative_other |   short_text |   url_obfuscation |
|:-------------|-----------------------:|-------------:|------------------:|
| FN           |                      1 |            1 |                 1 |

### TELEGRAM

| error_type   |   ambiguous_business |   char_substitution |   currency_symbols |   false_negative_other |   false_positive_other |   long_subtle_spam |   short_text |   template_marketing |
|:-------------|---------------------:|--------------------:|-------------------:|-----------------------:|-----------------------:|-------------------:|-------------:|---------------------:|
| FN           |                    0 |                   5 |                  0 |                     10 |                      0 |                  6 |            1 |                    2 |
| FP           |                    1 |                   1 |                  1 |                      0 |                     18 |                  0 |            4 |                    1 |

## Representative Error Examples

*Note: All examples have been anonymized to remove PII.*

### SMS

**false_negative_other** (FN: predicted ham, actual spam)
> Did you hear about the new "[NAME]"? It comes with all of Ken's stuff!

**short_text** (FN: predicted ham, actual spam)
> 2/2 146tf150p

**url_obfuscation** (FN: predicted ham, actual spam)
> Cashbin.co.uk (Get lots of cash this weekend!) [URL] [NAME] to the weekend We have got our biggest and best EVER cash give away!! These..

### TELEGRAM

**false_negative_other** (FN: predicted ham, actual spam)
> sms ac jsco energy is high but u may not know where 2channel it 2day ur leadership skills r strong psychic reply ans wquestion end reply end jsco

**false_negative_other** (FN: predicted ham, actual spam)
> 🚀 adx 10 profit so far and still counting 😍

**false_positive_other** (FP: predicted spam, actual ham)
> career opportunity dear mr kaminski i will forward my resume i am looking for a trading position i have three years of market making experience in illiquid markets which i beleive is highly relevant b...

**false_positive_other** (FP: predicted spam, actual ham)
> oin sini aja kak asik

**long_subtle_spam** (FN: predicted ham, actual spam)
> imuns are known for their quality high standards and difficulty levels and participating in one will work as an asset when you apply for colleges or jobs 🎖 top ivy league colleges such as yale and har...

**long_subtle_spam** (FN: predicted ham, actual spam)
> 𝑰 𝒔𝒂𝒘 𝒂 𝒑𝒐𝒔𝒕 𝒂𝒃𝒐𝒖𝒕 𝑴𝒓 𝑾𝒊𝒍𝒍𝒊𝒂𝒎𝒔 𝑾𝒊𝒍𝒇𝒓𝒆𝒅 𝒊 𝒅𝒆𝒄𝒊𝒅𝒆𝒅 𝒕𝒐 𝒈𝒊𝒗𝒘 𝒉𝒊𝒎 𝒂 𝒕𝒓𝒚 𝒂𝒏𝒅 𝒏𝒐𝒘 𝒉𝒆𝒓𝒆 𝒊 𝒂𝒎 𝒈𝒊𝒗𝒊𝒏𝒈 𝒕𝒆𝒔𝒕𝒊𝒎𝒐𝒏𝒚 𝒂𝒃𝒐𝒖𝒕 𝒉𝒊𝒎 𝒕𝒐𝒐 𝒃𝒆𝒄𝒂𝒖𝒔𝒆 𝒊 𝒆𝒂𝒓𝒏𝒆𝒅 𝒂 𝒉𝒖𝒈𝒆 𝒂𝒎𝒐𝒖𝒏𝒕 𝒊𝒏 𝒋𝒖𝒔𝒕 𝒇𝒆𝒘 𝒕𝒓𝒂𝒅𝒊𝒏𝒈 𝒘𝒆𝒆𝒌𝒔 𝒘𝒊𝒕𝒉 𝒉𝒊𝒎 𝒈𝒆𝒕 𝒊𝒏 𝒕𝒐𝒖𝒄𝒉 𝒊𝒇 y𝒐...

**short_text** (FP: predicted spam, actual ham)
> god help us all

**short_text** (FP: predicted spam, actual ham)
> harharhar

**template_marketing** (FN: predicted ham, actual spam)
> want to make women adore you click here 10 minutes before sex lasts for 24 36 hours an eye for an eye makes the whole world blind the man who runs may fight again where there is an open mind there wil...

**template_marketing** (FP: predicted spam, actual ham)
> your confirmation is needed please respond to energy news live daily update confirmation from lyris listmanager please reply to this email message to confirm your subscription to enl dailyupdate txt y...

**ambiguous_business** (FP: predicted spam, actual ham)
> this message is not supported by your version of telegram please update to the latest version in settings advanced or install it from

**currency_symbols** (FP: predicted spam, actual ham)
> sir i have been late in paying rent for the past few months and had to pay a ltgt charge i felt it would be inconsiderate of me to nag about something you give at great cost to yourself and thats why ...

**char_substitution** (FN: predicted ham, actual spam)
> its very hard to believe when my friend told me the secret behind his success i have seen it with my eyes and im very happy to be part of it contact the man that makes it possible for me you can conta...

**char_substitution** (FN: predicted ham, actual spam)
> 💎ғᴏʀᴍ : 💳sᴇʟʟᴇʀ : hikaruasahi 💰ᴘʀɪᴄᴇ : 2000 ⚔️ᴘᴏᴋᴇᴍᴏɴ ʟᴇᴠᴇʟ :57 🧬ɴᴀᴛᴜʀᴇ : jolly ✅sᴛᴀᴛᴜs :available

