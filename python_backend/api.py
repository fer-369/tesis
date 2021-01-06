from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from os.path import join, dirname, realpath
import mysql.connector
import pandas as pd
import json
from decimal import Decimal
import simplejson as json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from PIL import Image, ImageDraw, ImageFilter

app = Flask(__name__, static_url_path='/static')

dir = os.path.dirname(__file__)
UPLOAD_FOLDER =  os.path.join(dir, './static/files/')
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/coordinates/flush')
def truncateCoords():
    mydb = mysql.connector.connect(
        host="35.239.252.118",
        user="root",
        password="conta1.2",
        database="tesis_gis"
    )

    mycursor = mydb.cursor()

    mycursor.execute("TRUNCATE TABLE dron_capture;")

    response = jsonify("sucess")
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/coordinates')
def getCoordinates():
    mydb = mysql.connector.connect(
        host="35.239.252.118",
        user="root",
        password="conta1.2",
        database="tesis_gis"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM dron_capture")

    myresult = mycursor.fetchall()

    data = []

    for user in myresult:
        data.append(user)
   
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/idw')
def getIdw():
    mydb = mysql.connector.connect(
        host="35.239.252.118",
        user="root",
        password="conta1.2",
        database="tesis_gis"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT latitude FROM dron_capture")

    latitude = mycursor.fetchall()

    latFloat = []

    for temp in latitude:
        latFloat.append(float(temp[0]))

    mycursor.execute("SELECT longitude FROM dron_capture")

    longitude = mycursor.fetchall()

    longFloat = []

    for temp in longitude:
        longFloat.append(float(temp[0]))

    mycursor.execute("SELECT pressure FROM dron_capture")

    pressure = mycursor.fetchall()

    pressureFloat = []

    for temp in pressure:
        pressureFloat.append(float(temp[0]))

    mycursor.execute("SELECT temperature FROM dron_capture")

    temperature = mycursor.fetchall()

    temperatureFloat = []

    for temp in temperature:
        temperatureFloat.append(float(temp[0]))

    mycursor.execute("SELECT humidity FROM dron_capture")

    humidity = mycursor.fetchall()

    humidityFloat = []

    for temp in humidity:
        humidityFloat.append(float(temp[0]))

    # Setup: Generate data...
    n = 10
    nx, ny = 10, 10
    #x, y, z = map(np.random.random, [n, n, n])
    y = np.array(longFloat)
    x = np.array(latFloat)
    z = np.array(pressureFloat)
    z1 = np.array(temperatureFloat)
    z2 = np.array(humidityFloat)

     #print(x)
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    # Calculate IDW
    grid1 = simple_idw(x,y,z,xi,yi)
    print((grid1))
    grid1 = grid1.reshape((ny, nx))

    # Calculate scipy's RBF
    grid2 = simple_idw(x,y,z1,xi,yi)
    print((grid2))
    grid2 = grid2.reshape((ny, nx))

    grid3 = simple_idw(x,y,z2,xi,yi)
    print((grid1))
    grid3 = grid3.reshape((ny, nx))

    # Comparisons...
    plot(x,y,z,grid1, "pressure")

    plot(x,y,z,grid2, "temp")
 
    plot(x,y,z,grid3 , "humidity")
   
    response = jsonify("sucess")
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Get the uploaded files
@app.route("/upload", methods=['POST'])
def uploadFiles():
      # get the uploaded file
        mydb = mysql.connector.connect(
            host="35.239.252.118",
            user="root",
            password="conta1.2",
            database="tesis_gis"
        )   

        mycursor = mydb.cursor()
    
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
           # parseCSV(file_path)
           data = pd.read_csv(file_path, header = None)
           for index, row in data.iterrows():
                print((re.findall(r"[-+]?\d*\.\d+|\d+", str(row[2])))[0])
                sql = "INSERT INTO dron_capture (latitude, longitude, temperature, pressure, altitude, humidity) VALUES (%s, %s, %s, %s, %s, %s)"
                value = (re.findall(r"[-+]?\d*\.\d+|\d+", str(row[3]))[0], 
                    re.findall(r"[-+]?\d*\.\d+|\d+", str(row[2]))[0], 
                    re.findall(r"[-+]?\d*\.\d+|\d+", str(row[5]))[0], 
                    re.findall(r"[-+]?\d*\.\d+|\d+", str(row[7]))[0], 
                    re.findall(r"[-+]?\d*\.\d+|\d+", str(row[4]))[0], 
                    re.findall(r"[-+]?\d*\.\d+|\d+", str(row[6]))[0])
                mycursor.execute(sql, value)
                mydb.commit()
            # save the file
        return redirect(url_for('index'))

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi


def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)


def plot(x,y,z,grid, name):
    #plt.figure()
    #plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    #plt.hold(True)
    #plt.scatter(x,y,c=z)
    #plt.colorbar()
    plt.imsave('static/images/idw_' + name + '.png', grid, dpi=300)
    im_rgb = Image.open('static/images/idw_'+name+'.png')
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(164)
    im_rgba.resize((300,300))
    im_rgba.save('static/images/idw_' + name + '.png')

    length_x, width_y = im_rgba.size
    factor = min(5, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = im_rgba.resize(size, Image.ANTIALIAS)
    image_resize = image_resize.rotate(90)
    image_resize.save('static/images/idw_'+name+'.png', dpi=(300, 300))
