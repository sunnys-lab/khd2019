>#### Login
Github UserName(ID 아님)와 비밀번호로 로그인  
~~~
nsml login
sunnys-lab
dlrltjs00!

INFO[2018/11/19 12:31:40.032] connecting to hack-cli.nsml.navercorp.com:18553
INFO[2018/11/19 12:31:42.058] there is no update
GitHub Username: sunnys-lab
GitHub Password: **********
INFO[2018/11/19 12:33:42.355] Welcome to NSML!
~~~

>#### Run a session
~~~
nsml run -d ir_ph2 -e main.nasnet.mobile.py
nsml run -d KHD2019_FUNDUS -e main_org.py
~~~

>#### Session List
~~~
nsml ps -n 10
~~~

>#### Get a log from session
~~~
nsml logs -f Sunny/ir_ph1/26
~~~

>#### List models
~~~
nsml model ls Sunny/ir_ph2/79
~~~

>#### Submit a session
~~~
nsml submit Sunny/ir_ph2/86 11
nsml submit team144/KHD2019_FUNDUS/1 1
~~~

>#### Show leaderboard of dataset
~~~
nsml dataset board ir_ph1_v2
~~~

>#### Delete model
~~~
nsml model rm Sunny/ir_ph2/79 "*"

nsml model ls Sunny/ir_ph1_v2/7
~~~


>#### Change Session Memo
~~~
nsml memo Sunny/ir_ph2/373 "nasnet.ft3"
~~~

