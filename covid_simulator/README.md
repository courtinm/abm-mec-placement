To run this project you need to use python3. Once Python3 is set up on your machine, you can run the following commands to install virtualenv:
```
sudo apt-get install python3-venv
```

Then create your virtual environment:
```
python3 -m venv myenv
```

Activate the virutal environment:
```
source myenv/bin/activate
```

Install all required packages:
```
pip3 install -r requirements.txt
```

Then, simply run one of the examples from the scenarios folder by copying it in the current folder, or run the `base_scenario.py`:
```
python base_scenario.py
```
