# Created by Nuo Wang.
# Last modified on 8/16/2017.

# The Python libaries needed for this Flask app.
from flask import Flask, request, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons
import geocoder
import pandas as pd
import numpy as np

# Create the Flask object.
app = Flask(__name__, template_folder="templates")

# Set up the Google Maps API key.
app.config['GOOGLEMAPS_KEY'] = "_______________________________________"
GoogleMaps(app, key="_______________________________________")

# Load preprocessed data.
data = pd.read_csv("./data/all_doctors.csv")

# Initialize distance (to query map location) and doctor search columns.
data['dist'] = 0.0
data['is_input_doc'] = 1  # by default no doctor is filtered out, since no name search

# The code for the app.
@app.route('/', methods=['GET', 'POST'])
def index():
    # Default search name and address to none.
    doctor_name = ""
    address = ""
    
    # If a search (by name and/or by address) is requested.
    # Return a page with search results.
    if request.method == "POST":
        # Retrieve the input search terms.
        params = dict(request.form)
        doctor_name = params["ticker"][0]
        address = params["ticker"][1]
        location = geocoder.google(address).latlng
        
        # If there is a search address, calculate every doctor's distance to that address.
        if location != []:
            loc_in = np.array((location[0], location[1]))
            for index, row in data.iterrows():
                loc_t = np.array((row["loc_lat"], row["loc_lon"]))
                data.set_value(index, "dist", np.sum((loc_in - loc_t)**2)) # no need to take sqrt
        # If there is no search address, reset all distances (to address) to 0.
        else:
            for index, row in data.iterrows():
                data.set_value(index, "dist", 0.0)
        
        # If there is a search doctor name, set the name match for a doctor to True
        # if any of their first or last name contains any searched name splitted by space.
        if doctor_name != "":
            name_list = doctor_name.lower().split(" ")
            for index, row in data.iterrows():
                for element in name_list:
                    data.set_value(index, "is_input_doc", 1)
                    if row["first_name"].lower().find(element) == -1 and row["last_name"].lower().find(element) == -1:
                        data.set_value(index, "is_input_doc", 0)
        # If there is no search doctor name, set name match to True for all doctors.
        else:
            for index, row in data.iterrows():
                data.set_value(index, "is_input_doc", 1)
    
        # Sort the doctors in the dataframe first by distance then by name match.
        sorted_data = data.sort_values(["dist", "is_input_doc"], ascending=[1, 0]).reset_index(drop=True)
        
        # Initialize variables storing the search results.
        markers_in = []
        selected = []
        lat_in = 0
        lon_in = 0
        
        # Return search results only if at least one search term is given.
        if doctor_name != "" or location != []:
            counter = 0
            # Only return up to 10 top search results.
            for i in range(0,10):
                if sorted_data.loc[i]["is_input_doc"] == 1:
                    # Add doctor to search result list.
                    selected.append(sorted_data.loc[i])
                    markers_in.append({
                                       'icon': "static/images/l{0}s.png".format(str(counter)),
                                       'lat': sorted_data.loc[i]["loc_lat"] + np.random.random()*0.0001,
                                       'lng': sorted_data.loc[i]["loc_lon"] + np.random.random()*0.0001,
                                       'infobox': "{0} {1}, {2}".format(sorted_data.loc[i]["first_name"], sorted_data.loc[i]["last_name"], sorted_data.loc[i]["title"])
                                      })
                    counter += 1
                    
            # Update the map position to the location of the most relevant doctor.
            lat_in = sorted_data.loc[0]["loc_lat"]
            lon_in = sorted_data.loc[0]["loc_lon"]
        # If no search term is given, update the map position to default (the position of San Francisco).
        else:
            lat_in = 37.774507
            lon_in = -122.419255
        
        # The number of search results.
        length = len(selected)
        
        # Create map containing the search results.
        movingmap = Map(
            identifier="movingmap",
            varname="movingmap",
            style='height:400px;width:100%;margin:0;',
            lat=lat_in,
            lng=lon_in,
            markers=markers_in,
            zoom=13
        )
        # Return map.
        return render_template('index.html',
                               movingmap=movingmap,
                               doctor_name=doctor_name,
                               address=address,
                               selected=selected)

    # If this is the first time loading the page or the search terms are empty.
    # Return the default page.
    movingmap = Map(
        identifier="movingmap",
        varname="movingmap",
        style='height:400px;width:100%;margin:0;',
        lat=37.774507,
        lng=-122.419255,
        zoom=12
    )
    selected = []
    return render_template('index.html',
                           movingmap=movingmap,
                           doctor_name=doctor_name,
                           address="367B 9th St",
                           selected=selected)

# Run app from here.
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
