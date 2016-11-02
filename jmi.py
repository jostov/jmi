import Tkinter as tk
import os, time, pygame, math, importlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from myo_raw import MyoRaw as mraw

#of the currently available classifiers. Simply add an entry to this
#dictionary to add a new classifier to this program

#In order for a classifier to work with this program, it simply needs
#to have fit() and predict() methods with inputs similar to those implemented
#in the following classes
classifier_dict = { 'Ada' : 'sklearn.ensemble.AdaBoostClassifier',
                  'GNB' : 'sklearn.naive_bayes.GaussianNB',
                  'RF' : 'sklearn.ensemble.RandomForestClassifier',
                  'GradBoost' : 'sklearn.ensemble.GradientBoostingClassifier',
                  'SVC' : 'sklearn.svm.SVC',
                  'KNN' : 'sklearn.neighbors.KNeighborsClassifier',
                  'LDA' : 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'}

#Handles dynamic imports
for each in classifier_dict:
  try:
    b = importlib.import_module(''.join([j + '.' for j in classifier_dict[each].split('.')[:-1]])[:-1])
    classifier_dict[each] = getattr(b, classifier_dict[each].split('.')[-1])
  except ImportError:
    print each + " failed to import"


class HideousAbomniination(object):
	def __init__(self, nomyo=False):
		self.instructions = []
		self.ohgodwhatamidoingitsfourinthemorning = []
		if nomyo==False:
			self.m = mraw()

	  #Extremely important variable
		self.will_to_live = 0
		self.connected = False

		#Recording flags
		self.recording = False
		self.predicting = False
		self.rollover = 0
		self.window_length = 10
		self.recsamps = 50
		self.rec = 0

		#Data collection areas
		self.data = {'vectors' : [], 'labels' : []}
		self.freshdata = []
		self.current_class = None
		self.feature_vector = None

		self.classyfire = RandomForestClassifier()

	def connect_damnit(self):
		print "Connecting and shit"
		self.m.add_emg_handler(self.emg_shit)
		self.m.connect()
		self.connected = True
		print "Connected and shit"

	def add_new_handler(self, handler):
		self.ohgodwhatamidoingitsfourinthemorning.append(handler)

	def set_current_class(self, cl):
		self.current_class = cl

	def set_classifier(self, cls):
		self.classyfire = cls()

	def set_recording(self):
		self.recording = True 

	def set_no_recording(self):
		self.recording = False

	def set_predicting(self):
		self.predicting = True if self.predicting is False else False
		
	def train(self):
		print 'training classifier'
		self.classyfire.fit(self.data['vectors'], self.data['labels'])
		self.predicting = True

	def predict(self, vector):
		self.last_prediction = str(self.classyfire.predict(vector)) + 'made at '+ str(time.asctime())
		print "["+ str(time.asctime())+"]" + str(self.last_prediction)

	def feature_calc(self, Dataset, Nframe, window_length):
		#Various parameters set by the tdfeats.m
		DEADZONE_ZC = .025
		DEADZONE_TURN = .015
		SCALE_ZC = 15
		SCALE_MAV = 2
		ruler = 1/ window_length
		rulrsq = ruler ** 2
		lscale = window_length / 40.0
		tscale = (window_length / 40.0) * 10


		new_data = np.array(Dataset)
		Ntotal, Nsig = np.shape(new_data)
		new_data = new_data - np.mean(new_data, axis=0) 
		feature_vector = np.zeros((4,Nsig))
		m,z,t,w = [0,3,2,1]
		feature_vector[m,:] = np.mean(np.absolute(new_data),axis=0)
		feature_vector[z,:] = np.zeros(Nsig)
		feature_vector[t,:] = np.zeros(Nsig)
		feature_vector[w,:] = np.zeros(Nsig)

		absolutes = np.absolute(Dataset)
		for i in range(Nsig):
			feature_vector[z,i] = (np.diff(np.sign(new_data[:,i])) !=0).sum()
		for j in range(1,Ntotal-1):
			feature_vector[w,:] = feature_vector[w,:] + np.sqrt((((absolutes[j-1,:] - absolutes[j,:]))/20.0)**2 + rulrsq)
			for i in range(Nsig):
				if ((new_data[j-1,i] < new_data[j,i]) and (new_data[j,i] > new_data[j+1,i])) or ((new_data[j-1,i] > new_data[j,i]) and (new_data[j,i] < new_data[j+1,i])):
					if ((absolutes[j,i]-absolutes[j-1,i] > DEADZONE_TURN)) or (absolutes[j,i] - absolutes[j+1,i] > DEADZONE_TURN):
						feature_vector[t,i] +=1

		feature_vector[z,:] = feature_vector[z,:] / SCALE_ZC * 40 / window_length
		feature_vector[m,:] = feature_vector[m,:] / SCALE_MAV
		feature_vector[w,:] = (feature_vector[w,:]- 1) / lscale
		feature_vector[t,:] = feature_vector[t,:]/ tscale
		return feature_vector.flatten('F')

	def calc_features(self):
		self.feature_vector = self.feature_calc(np.array(self.freshdata), 1, 40)
		if self.recording is True:
			print 'added feature vector to training list'
			self.data['vectors'].append(self.feature_vector)
			self.data['labels'].append(self.current_class)
			self.rec += 1
			if self.rec >= self.recsamps:
				print "finished recording " + self.current_class
				self.recording = False
				self.rec = 0
		if self.predicting:
			self.predict(self.feature_vector)

	def emg_shit(self, emg, moving, times=[]):
		self.freshdata.append(emg)
		if len(self.freshdata) >= self.window_length + int(math.floor(self.window_length * (1.0 - self.rollover))) :
			self.calc_features()
			self.freshdata = self.freshdata[int(math.floor(self.window_length * self.rollover)):]

		#In case something went horribly wrong in my logic.
		elif len(self.freshdata) > 2 * self.window_length: 
			print "throwing out " + str(len(self.freshdata)) + " unsynced samples"
			self.freshdata = []

	#This adds instructions to the instruciton queue
	def instruct_queue(self, instruction):
		self.instructions.append(instruction)
	
	#Refresh this, always. It handles instructions.
	def advance_queue(self):
		if len(self.instructions) > 0:
			self.instructions.pop()()

	#Sets the number of samples taken per training session.
	def set_sample_count(self, count):
		self.recsamps=count
		
	#Changes rate of conversion from raw data to feature data
	def set_window_length(self, length):
		self.window_length=length

	#Sets the rollover amount
	def set_rollover_percent(self, rollover):
		self.rollover = rollover

	def is_predicting(self):
		return self.predicting


####Fun-ctions####

def add_class_name():
	class_list.insert(tk.END, class_entry.get())
	class_entry.delete(0,tk.END)

def train_flasher(countdown=2, last_dec=None, font=None, on_finish=None):
	if font is None:
		font = pygame.font.Font(None, 100)
	t = time.time()
	if last_dec is None:
		screen.fill((255,255,255))
		last_dec = t
		b = font.render(str(countdown), 1, (5,5,5), (255,255,255))
		screen.blit(b, (0,0))
		c.instruct_queue(lambda : train_flasher(countdown, t, font, on_finish))
	elif t >= last_dec + 1 and countdown > 0:
		screen.fill((255,255,255))
		count = countdown - 1
		b = font.render(str(count), 1, (5,5,5), (255,255,255))
		screen.blit(b, (0,0))
		c.instruct_queue(lambda : train_flasher(count, t, font, on_finish))
	elif countdown == 0:
		screen.fill((255,255,255))
		c.instruct_queue(lambda : on_finish())
	else:
		c.instruct_queue(lambda : train_flasher(countdown, last_dec, font, on_finish))

def train_selected_class():
	c.set_sample_count(int(samples_box.get()))
	c.set_window_length(int(window_box.get()))
	c.set_rollover_percent(float(rollover_quantity.get()))
	c.set_current_class(class_list.get(tk.ACTIVE))
	train_flasher(on_finish=start_training)

def set_classifier():
	print classifier_pick.get()
	c.set_classifier(classifier_dict[classifier_pick.get()])

def start_predicting():
	if c.is_predicting():
		c.predict()
		c.instruct_queue(lambda : start_predicting)

#This is some crazy sorcery from the myo_raw, I'm not so sure adding this
#was a great idea, but well, here it goes anyways. forced dzhu collab, ya mean?
last_vals = None
def plot(vals):
	w = 1000
	h = 200
	global last_vals
	if last_vals is None:
		last_vals = vals
		return
	D = 5
	screen.scroll(-D)
	screen.fill((0,0,0), (w - D, 0, w, h))
	for i, (u, v) in enumerate(zip(last_vals, vals)):
		c = int(255 * max(0, min(1, v)))
		screen.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8));
	pygame.display.flip()
	last_vals = vals

#don't graph shit yet. you're not ready.

def start_training():
	print "got called at least"
	c.set_recording()

def myo_connect():
	c.connect_damnit()
	#start_graphing()

####Start of the shitshow####

#Root tk object
c = HideousAbomniination()
root = tk.Tk()
#Embedded frame for pygame
embed = tk.Frame(root, width=200, height=500)
embed.grid(columnspan = (200), rowspan = 500)
embed.pack(side = tk.LEFT)
#Button container 
buttonwin = tk.Frame(root, width=300, height = 500)
buttonwin.pack(side=tk.LEFT)
print embed.winfo_id()
os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
root.update()
screen = pygame.display.set_mode((200,500))
screen.fill(pygame.Color(255,255,255))
#scr = pygame.Surface((1000,300))
#screen.blit(scr, (0,200))

#text_display = pygame.Surface((500,100))
#text_display.blit(screen, (0,0))
pygame.display.init()
pygame.display.update()
pygame.font.init()

# Buttons being made or some shit
classifier_box = tk.Frame(buttonwin)
classifier_box.grid(row=0)
classifier_pick = tk.StringVar(classifier_box)
classifier_pick.set('RF')
classifier_entry = apply(tk.OptionMenu, (classifier_box, classifier_pick) + tuple(classifier_dict.keys()))
classifier_entry.grid(row=0, column=0)
classifier_button = tk.Button(classifier_box, text='Set Classifier', command = set_classifier)
classifier_button.grid(row=0, column=1)
class_list = tk.Listbox(buttonwin)
class_list.grid(row=1,column=0)
class_entry = tk.Entry(buttonwin)
class_entry.grid(row=2,column=0)
list_del = tk.Button(buttonwin, text="Delete",
		           command=lambda lb=class_list: lb.delete(tk.ANCHOR)).grid(row=3,column=0)
list_add = tk.Button(buttonwin, text='Add item', command = add_class_name).grid(row=4,column=0)
#These objects are all None, but if they need to be referred to later, the names
#are already reserved
myo_button = tk.Button(buttonwin, text='Connect to Myo', command=myo_connect).grid(row=5,column=1)
training_button = tk.Button(buttonwin, text='Training session', command=train_selected_class).grid(row=2,column=1)
active_button = tk.Button(buttonwin, text='Train Classifier', command=c.train).grid(row=3,column=1)
active_button = tk.Button(buttonwin, text='Toggle Classifier', command=c.set_predicting).grid(row=4,column=1)

dial_box= tk.Frame(buttonwin)
dial_box.grid(row=1, column=1)
countdown_box = tk.Spinbox(dial_box, width=2, from_=0, to=99)
countdown_box.delete(0,tk.END) 
countdown_box.insert(0, 2) 
countdown_box.grid(row=0, column=0)
countdown_box.delete(0, tk.END)
countdown_box.insert(0,2)
cdown_text = tk.Label(dial_box, text='Countdown length')
cdown_text.grid(row=0, column=1)

#Sample number box
samples_box = tk.Spinbox(dial_box, width=2, from_=0, to=99)
samples_box.grid(row=1, column=0)
samples_box.delete(0,tk.END)
samples_box.insert(0, 50)
sample_label = tk.Label(dial_box, text='Sample number')
sample_label.grid(row=1, column=1)

#Rate at which feature vectors are packed
window_box= tk.Spinbox(dial_box, width=2, from_=1, to=99)
window_box.delete(0,tk.END)
window_box.insert(0, 10)
window_box.grid(row=2, column=0)
window_length = tk.Label(dial_box, text='Window length')
window_length.grid(row=2, column=1)

#Some other spinbox
rollover_quantity = tk.Spinbox(dial_box, width=2, increment=0.01, from_=0.0, to=1)
rollover_quantity.delete(0,tk.END)
rollover_quantity.insert(0, .2)
rollover_quantity.grid(row=3, column=0)
rollover_label = tk.Label(dial_box, text='Rollover Quantity')
rollover_label.grid(row=3, column=1)

root.update()



while True:
	if c.connected is True:
		c.m.run(1)
	c.advance_queue()
	pygame.display.update()
	root.update()
