import requests

city = input('input the city name')
print(city)
city = 'kerala'
print('Displaying Weather report for: ' + city)

url = 'https://wttr.in/{}'.format(city)
res = requests.get(url)
print(res.text)