#wirtten by cwf(cwfsdtc163@163.com)

#The Question
# How can i get url from clipboard
# need a GUI
# get all picture from a alubm

import urllib.request
import urllib.parse
import urllib.error
import re
import os
import http.cookiejar

def bcy_login():
    LOGIN_URL = 'https://bcy.net/public/dologin'#you should use fiddler+firefox to confirm this URL,if you use bcy.net/login here you will not success!!!(because you can't get 'LOGGED_USER' from cookie)
    values = {'email':'you_email', 'password':'you_pass','remember': '1'}#the key-value here is different with other WEB site!!
    postdata = urllib.parse.urlencode(values).encode()

    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'
    headers = {'User-Agent': user_agent, 'Connection': 'keep-alive'}

    cookie_filename = 'f:\\bcy_pictures\\cookie.txt'
    cookie = http.cookiejar.MozillaCookieJar(cookie_filename)
    handler = urllib.request.HTTPCookieProcessor(cookie)
    opener_cookie = urllib.request.build_opener(handler)


    request = urllib.request.Request(LOGIN_URL, postdata, headers)
    try:
        response = opener_cookie.open(request)
        page = response.read().decode('utf-8')
        #print(page)
    except urllib.error.URLError as e:
        print(e.code, ':', e.reason)

    cookie.save(ignore_discard=True, ignore_expires=True)#save cookie

    return opener_cookie#return a opener with cookie



def getHtml(url, opener):
    #make the request like a browser
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36'}

    #load cookie
    cookie = http.cookiejar.MozillaCookieJar()
    cookie.load('f:\\bcy_pictures\cookie.txt', ignore_discard=True, ignore_expires=True)

    req = urllib.request.Request(url, headers=headers)
    #response = urllib.request.urlopen(req)
    response = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie)).open(req)
    html = response.read().decode('utf-8')
 
    #print(html)
    return html

def getPicture(html):

    re_pattern=re.compile(r'https://img[0-9].bcyimg.com/coser/[0-9]*/post/[a-z0-9A-Z]*/[a-z0-9A-Z]{1,32}\.jp[e]{0,1}g/w650')
    pictures = re_pattern.findall(html)
    return pictures

def savePictures(PictureList, DirPath):
    
    for item  in PictureList:
        re_pic_name=re.compile(r'https://img[0-9].bcyimg.com/coser/[0-9]*/post/[a-z0-9A-Z]*/([a-z0-9A-Z]{1,32})\.jp[e]{0,1}g/w650')
        pic_name = re_pic_name.findall(str(item))
        
        dest_file = str.format('{}\\{}.jpg',DirPath,str(pic_name)[2:-2])#delete the "["and "'" and make the full path include file name
        print(dest_file)
        urllib.request.urlretrieve(item[0:-5], dest_file)#when urlretrieve write file, if the file exits, it will be overwritten!
        
    return

def makeDir(html):

    #get all title
    re_ptn_title=re.compile(r'<meta name="keywords" content="(.*)"')
    all_title = str(re_ptn_title.findall(html))[2:-2]#delete the "["and "'"
    #print(all_title)

    #replace the char can't be in dirname(:?<>*\/→)
    re_ptn_delchar=re.compile('[\:\?\<\>\*\\\/→]')
    all_title = re_ptn_delchar.sub('a',all_title)
    #you also can use str.replace like "titiles[0].replace(':', '1')"
    print(all_title)

    
    #splilt title
    titiles = str(all_title).split(',')
    
    
    #get dir name 'coser_title1_title2'
    dir_name = str.format('{}_{}_{}', titiles[0], titiles[1], titiles[2])
    dir_full_name = str.format('{}{}', 'f:\\bcy_pictures\\', dir_name)
    #print(dir_name)

    #make dir
    try:
        os.makedirs(dir_full_name)
    except FileExistsError:
        #do nothing(but print message)
        print('dir {} exist, continue\n'.format(dir_full_name))
    except:
        print('mkdir err\n')
        dir_full_name = 'f:\\bcy_pictures'#use the base path
    else:
        #success do nothing
        #print('save in:{}\n'.format(dir_full_name))
        pass
   
    return dir_full_name

def getUrlFromClipBoard():
    
    return

#----------------------------------------------------------------------
#main func

html = getHtml("https://bcy.net/coser/detail/112151/1243639", bcy_login())#this URL is a COS for iron man
#problem
#nothing now :)

picture_list = getPicture(html)

savePictures(picture_list, makeDir(html))

#print(html)
