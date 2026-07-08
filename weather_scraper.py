import requests
import pandas as pd
import matplotlib.pyplot as plt


API_KEY = "YOUR_API_KEY"

city = input("Enter City Name: ")

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    city_name = data["name"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind_speed = data["wind"]["speed"]
    weather = data["weather"][0]["description"]

    print("\n------ Weather Report ------")
    print("City        :", city_name)
    print("Temperature :", temperature, "°C")
    print("Humidity    :", humidity, "%")
    print("Pressure    :", pressure, "hPa")
    print("Wind Speed  :", wind_speed, "m/s")
    print("Condition   :", weather)

    
    weather_data = {
        "City": [city_name],
        "Temperature (°C)": [temperature],
        "Humidity (%)": [humidity],
        "Pressure (hPa)": [pressure],
        "Wind Speed (m/s)": [wind_speed],
        "Condition": [weather]
    }

    df = pd.DataFrame(weather_data)

    
    df.to_csv("weather_report.csv", index=False)

    print("\nWeather report saved as weather_report.csv")

    # Temperature Chart
    plt.figure(figsize=(5,4))
    plt.bar(["Temperature"], [temperature])
    plt.title("Temperature")
    plt.ylabel("°C")
    plt.show()

    
    plt.figure(figsize=(5,4))
    plt.bar(["Humidity"], [humidity])
    plt.title("Humidity")
    plt.ylabel("%")
    plt.show()

else:
    print("City not found or Invalid API Key.")
