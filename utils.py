def parseData(data: dict) -> str:
    # print(data.values())
    name = list(data.keys())[0]
    hotel = data[name]

    text_format = (
        f"\tHotel name: {name}\n"
        f"City {hotel['city']}\n"
        f"Price per night:{hotel['price_per_night']}\n"
        f"Rating: {hotel['rating']}\n"
        f"Review: {hotel['review']}\n"
    )
    return text_format
