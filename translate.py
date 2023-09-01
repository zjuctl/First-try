import random
import hashlib
import urllib
import requests
import json                                               #安装相应的库

def trans(word):
    src = 'zh'                                                #翻译的源语言
    obj = 'en'                                                #翻译的目标语言
    appid = '20230720001750976'                                     #这里输入你注册后得到的appid
    secretKey = 'oxgnI0cwLVhyirpSUTQL'                                  #这里输入你注册后得到的密匙

    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'                  #必须加上的头
    # word= input('请输入你要翻译的中文：')                                           #输入你要翻译的中文
    salt = random.randint(31256, 66253)                                           #产生随计数

    sign = appid + word + str(salt) + secretKey                                   #文档的step1拼接字符串
    m1 = hashlib.md5()
    m1.update(sign.encode('utf-8'))
    sign = m1.hexdigest()                                                         #文档的step2计算签名
    myur1 = myurl  + '?q=' + urllib.parse.quote(word) + '&from=' + src + '&to=' + obj + '&appid='+ appid + '&salt=' + str(salt) + '&sign=' + sign

    english_data = requests.get(myur1)                                            #请求url
    js_data = json.loads(english_data.text)                                       #下载json数据
    content = js_data['trans_result'][0]['dst']                                   #提取json数据里面的dst
    return content                                                             #打印出翻译的英文
