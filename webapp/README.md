# Instructions for running/deploying the DoctorSnapshot webapp
The DoctorSnapshot webapp is [Python Flask](http://flask.pocoo.org/) based, to run the DoctorSnapshot webapp, you will need to download this entire directory first and put in your [Google Maps API key](https://developers.google.com/maps/documentation/javascript/get-api-key) on lines 16 and 17 of `app.py`. Then you can run the app locally or deploy it to a server. An example of deploying to Heroku is given here.

## 1. Running the DoctorSnapshot app locally
Run `app.py`:
```
python app.py
```
Then open the app at `http://localhost:5000`.

## 2. Deploying DoctorSnapshot to Heroku
You can use the following commands to deploy your copy of DoctorSnapshot to your Heroku account.

### Create app
```
heroku create doctorsnapshot
git init
```

### Pull from Heroku
```
git pull https://git.heroku.com/doctorsnapshot.git
```

### Upload/update app
```
git add .
git commit -m "[commit comment]"
git push heroku master
```

### Open the Heroku app
```
heroku open
```

### App settings
Check git repository names
```
git remote -v
```

Check how many dynos are running
```
heroku ps
```

Set up how many dynos to run
```
heroku ps:scale web=1
```

Check logs
```
heroku logs --tail
```
