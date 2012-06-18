# -*- coding: utf-8 -*-

import threading
import multiprocessing
import uuid
import time
import random
import numpy

class DummyRecoLearner(object):
	
	def __init__(self):
		pass
	
	def train(self, dataset, sleeptime=0.500):
		"""
		Permet l'entraînement du modèle
		"""
		time.sleep(sleeptime)
	
	def predict(self, dataset, sleeptime=0.500):
		"""
		Permet de réaliser des recommendations à partir du model entrainé
		"""
		time.sleep(sleeptime)
	
	def evaluate(self, dataset, sleeptime=0.500):
		"""
		Permet d'évaluer un système de recommendation à partir du model entrainé
		"""
		time.sleep(sleeptime)

class DummyLearner(object):
	"""
	Ceci est une classe qui définie l'interface que doivent posséder
	les modèles.
	"""
	
	def __init__(self):
		pass
	
	def train(self, dataset, sleeptime = 0.500):
		"""
		Permet l'entraînement du modèle
		"""
		time.sleep(sleeptime)
	
	def predict(self, dataset, sleeptime = 0.200):
		"""
		Permet de réaliser des prédictions à partir du model entrainé
		"""
		time.sleep(sleeptime)


class SplitTrainVal():
	"""
	Cette classe split le dataset en un training set et un validation set respectant le testRatio
	"""
	
	def __init__(self, dataset, testRatio=0.25):
		self.testRatio = testRatio
		
		self.train = DataSet()
		self.test = DataSet()
		
		self.train.matrix = dataset[:round(dataset.shape[0]*(1-self.testRatio)), :]
		self.test.matrix = dataset[round(dataset.shape[0]*self.testRatio):,:]
	
	def __len__(self):
		return 1


class Folds():
	"""
	Cette classe permet de générer les folds à partir d'un dataset. Il sert principalement d'iterateur
	renvoyant
	"""
	def __init__(self, dataset, number):

		self.number = number
		
		if dataset.matrix.ndim == 1 or dataset.matrix.shape[0] < number:
			raise Exception("LE dataset doit comporter au moins autant d'observations que le nombre de folds")
			
		if (dataset.matrix.shape[0] % number) != 0:
			raise Exception("LE nombre de folds choisi doit être un multiple du nombre d'observations")
		
		self.folds = [ DataSet() for i in range(number)]
		
		for i in range(dataset.shape[0]): #Je parcours les lignes du dataset
			# Tout va être concatene en un seul grand vecteur qu'il faut redimensionné ensuite
			self.folds[i%number].matrix = numpy.concatenate((self.folds[i%number].matrix, dataset.matrix[i,:])) 
		
		for ds in self.folds:
			ds.matrix = ds.matrix.reshape(dataset.matrix.shape[0] / number, dataset.matrix.shape[1])
			ds.targetIndex = dataset.targetIndex


	def __len__(self):
		return self.number
	
	def __iter__(self):
		return self.forward()
		
	def __getitem__(self, index):
		# Todo: vérifier que index ne soit pas trop grand !
		return self.folds[index]
		
	
	def forward(self):
		# Todo: à remplacer par un for ?
		current_item = 0
		while(current_item < len(self.folds)):
			current_fold = self.folds[current_item]
			current_item+=1
			yield current_fold
			
	
	def reverse(self):
		# Todo: à remplacer par un for ?
		current_item = len(self.folds)-1
		while(current_item >= 0):
			current_fold = self.folds[current_item]
			current_item-=1
			yield current_fold
		
		

class DataSet(object):
	
	def __init__(self):
		self.matrix = numpy.array([]) #Empty array
		self.targetIndex = None
	
	def loadCSV(self, path, separator= ','):
		self.matrix= numpy.fromfile(path, separator, dtype= float)
		self.targetIndex = None
	
	def getTargets(self):
		if self.targetIndex is None:
			raise Exception("No target index define.")
		return self.matrix[:,self.targetIndex]
	
	def setTarget(self, index):
		if (index < self.shape[1]) and (index >= 0):
			self.targetIndex = index
		else:
			raise IndexError(index)
	
	def getObservations(self):
		observations = None
		for index in range(self.matrix.shape[1]):
			if index != self.targetIndex:
				if observations == None:
					observations = self.matrix[:,index].reshape(self.matrix.shape[0],1)
				else :
					observations = numpy.concatenate((observations, self.matrix[:,index].reshape(self.matrix.shape[0],1)),axis=1)
		
		return observations
	
	def folds(self, nb= 10):
		folds = [ numpy.array([]) for i in nb]
		for index in self.matrix.shape[0]:
			folds[index%nb] = numpy.concatenate(folds[index%nb], self.matrix[index, :])
		
		return folds
	
	@property	
	def shape(self):
		return self.matrix.shape
			

class MFJob(object):
	"""
	Encapsule un modèle ainsi que son uid (identifiant).
	"""
	def __init__(self, model, dataset ):
		self.model = model
		self.dataset = dataset
		self.uuid = str(uuid.uuid4())
		self.exception = None

class MFJobScikitsLearn(MFJob):
	"""
	Encapsule un modèle ainsi que son uid (identifiant).
	"""
	def __init__(self, model, X, Y):
		self.model = model
		self.X = X
		self.Y = Y
		self.uuid = str(uuid.uuid4())
		self.exception = None


class MFJobReco(MFJob):
	"""
	Encapsule un modèle, un dataset et un evaluator
	"""
	def __init__(self, model, evaluator):
		self.model = model
		self.evaluator = evaluator
		self.uuid = str(uuid.uuid4())
		self.exception = None

class WorkerError(Exception):
	"""
	Encapsule une exception qui apparaitrait dans un worker.
	"""
	def __init__(self, innerException):
		self.innerException = innerException
	
	def __str__(self):
		return "An exception occured in the worker."

# Todo: changer le nom car ca n'a rien à voir
class ThreadSafeContainer(object):
	"""
	Cette classe encapsule un dictionnaire ou une liste de manière ThreadSafe.
	Elle évite d'avoir à faire appel à Lock dans le code puisqu'il est intégré à cette classe-ci.
	"""
	
	def __init__(self, container= list()):
		self.container = container
		self.lock = threading.Lock()
	
	def __len__(self):
		with self.lock:
			return len(self.container)
	
	def append(self, value):
		with self.lock:
			self.container.append(value)
	
	def _lock(self):
		"""
		Parfois j'ai besoin de verrouiller manuellement.
		"""
		self.lock.acquire()
	
	def _release(self):
		self.lock.release()
	
	def __getitem__(self, obj):
		with self.lock:
			return self.container[obj]
	
	def __setitem__(self, obj, value):
		with self.lock:
			self.container[obj] = value
	
	def __delitem__(self, obj):
		with self.lock:
			del self.container[obj]
		
			


class ModelFactory(object):
	"""
	Une classe permettant l'entraînement de plusieurs modèles.
	Cet entraîneement est parallélisé et réponds à certaines contraites
	tel qu'un nombre maxmimum d'entraînement à la fois.
	"""

	
	def __init__(self, verbose = False):
		self.threadNbSem = threading.Semaphore(1)
		self.workQueue = multiprocessing.Queue()
		self.resultQueue = multiprocessing.Queue()
		self.resultDict = ThreadSafeContainer(dict()) # Used to store trained models
		self.eventDict = ThreadSafeContainer(dict())  # Used to wake up a particular blocked call to get()
		self.eventStop = multiprocessing.Event()
		self.runningJobs = ThreadSafeContainer(dict())
		
		self.verbose = False
		
		self._spooler = None
		self._listener = None
		
		
	
	def start(self):
		"""
		Lance l'entraînement des modèles dans la file.
		"""
		if self.verbose: print "Starting the model factory"
		self._listener = threading.Thread(target= self._listenerd)
		self._spooler = threading.Thread(target= self._spoolerd)
		self._listener.start()
		self._spooler.start()
		
	
	def stop(self, force= True):
		"""
		Mets un terme au lancement des entraînements, toutefois les modèles en cours d'entraînement
		ne sont pas arrêté. Force n'est pas encore implémenté!
		"""
		if self.verbose: print "Stopping the model factory"
		self.eventStop.set()
		self.workQueue.put(None) #Waking up spoolerd
		self.resultQueue.put(None) #Waking up listenerd
		self.threadNbSem.release() #Wake up spooler if waiting for Sem
		
		# Wake up all blocking call to get
		self.eventDict._lock()
		for key in self.eventDict.container:
			self.eventDict.container[key].set()
		self.eventDict._release()
	
	def add(self, model, dataset):
		"""
		Permet d'ajouter un nouveau modèle à entraîner. Cette fonction est asynchrone (ie. fonctionne même si la méthode .start() a été appelée).
		Renvoi un ticket qui permet de récupérer le model une fois qu'il a été entrainé
		"""
		newJob = MFJob(model, dataset)
		self.eventDict[newJob.uuid] = threading.Event()
		self.workQueue.put(newJob)
		if self.verbose: print "Added job to model factory (%s)" % newJob.uuid
		return newJob.uuid
	
	def get(self, uuid, delete= True):
		"""
		Permet de récupèrer un modèle entraîné. Cet appel est bloquant le temps que le modèle en question ai été entraîné.
		Si delete est à True (par defaut) alors seul un seul appel à get fonctionnera les autres leverons une exception
		"""
		if self.verbose: print "Waiting to get back job %s" % uuid
		
		try:
			event = self.eventDict[uuid]
		except IndexError:
			raise IndexError("Uuid (%s) does not exists" % uuid)
		
		event.wait()
		#Todo vérifier que l'on a pas été réveiller par le stop (eventStop)
		if self.eventStop.is_set():
			raise Exception("La mf a ete stoppé")

			

		
		job = self.resultDict[uuid]
		
		#Verifie si une exception a été levée dans le worker (process d'entraînement)
		if job.exception is not None:
			raise WorkerError(job.exception)
		
		
		if delete:
			del self.eventDict[uuid]
			del self.resultDict[uuid]
		
		if self.verbose: print "Got back job %s" % uuid
		
		return job
	
	def _spoolerd(self):
		"""
		Spooler qui se charge de lancer les job qui lui sont fourni à travers la workQueue.
		Ce n'est pas strictement nécessaire, on pourrait juste lancer X process qui irait chercher dans la queue
		du travail mais l'utilisation du process permet d'avoir un peu plus de contrôle sur l'ordre
		"""
		if self.verbose: print "Spoolerd: started"
		
		while True:
			if self.verbose: print "Spoolerd: Waiting for semaphore"
			self.threadNbSem.acquire()
			job = self.workQueue.get()
			if (self.eventStop.is_set()):
				#since the job will not be processed we put it back in the workQueue
				if not ( job is None):
					self.workQueue.put(job)
				return #Time to stop !
			
			if not (job is None): #When stopping we put None in the workQueue
				process = multiprocessing.Process(target=self._worker, args=(job, self.verbose))
				process.start()
				self.runningJobs[job.uuid] = process
				if self.verbose: print "Spoolerd: Job spooled %s" % job.uuid
			
	
	
	def _listenerd(self):
		"""
		Daemon chargé de récupèrer les modèles un fois l'entraînement terminé.
		Il se charge également de réveiller les appels .get() bloqué.
		"""
		if self.verbose: print "Listernd: started"
		while True:
			job = self.resultQueue.get()
			self.threadNbSem.release()
			
			if not (job is None): #When stopping we put None in the resultQueue
				self.resultDict[job.uuid] = job
				self.eventDict[job.uuid].set()
				self.runningJobs[job.uuid].terminate()
				del self.runningJobs[job.uuid]
				
				if self.verbose: print "Listerned: Got back job %s" % job.uuid
			
			if self.eventStop.is_set():
				return
	
	def _worker(self, job, verbose = False):
		"""
		Unité de travail pour l'entraînement. Une fois l'entraînement terminé, le modèle et renvoyé dans une queue.
		"""
		if self.verbose: print "worker: started job %s "% job.uuid
		try:
			job.model.train(job.dataset)
		except Exception, e:
			job.exception = e
		
		self.resultQueue.put(job)
		if self.verbose: print "worker: finished training for %s" % job.uuid

class ModelFactoryScikitsLearn(ModelFactory):
	
	def add(self, model, X, Y):
		"""
		Permet d'ajouter un nouveau modèle à entraîner. Cette fonction est asynchrone (ie. fonctionne même si la méthode .start() a été appelée).
		Renvoi un ticket qui permet de récupérer le model une fois qu'il a été entrainé
		"""
		newJob = MFJobScikitsLearn(model, X, Y)
		self.eventDict[newJob.uuid] = threading.Event()
		self.workQueue.put(newJob)
		if self.verbose: print "Added job to model factory (%s)" % newJob.uuid
		return newJob.uuid
	
	def _worker(self, job, verbose = False):
		"""
		Unité de travail pour l'entraînement. Une fois l'entraînement terminé, le modèle et renvoyé dans une queue.
		"""
		if self.verbose: print "worker: started job %s "% job.uuid
		try:
			job.model.fit(job.X, job.Y)
		except Exception, e:
			job.exception = e
		
		self.resultQueue.put(job)
		if self.verbose: print "worker: finished training for %s" % job.uuid

class ModelFactoryReco(ModelFactory):
	
	def add(self, model, evaluator):
		"""
		Permet d'ajouter un nouveau modèle à entraîner. Cette fonction est asynchrone (ie. fonctionne même si la méthode .start() a été appelée).
		Renvoi un ticket qui permet de récupérer le model une fois qu'il a été entrainé
		"""
		newJob = MFJobReco(model, evaluator)
		self.eventDict[newJob.uuid] = threading.Event()
		self.workQueue.put(newJob)
		if self.verbose: print "Added job to model factory (%s)" % newJob.uuid
		return newJob.uuid
	
	def _worker(self, job, verbose = False):
		"""
		Unité de travail pour l'entrainement et l'évaluation des méthodes de recommendation
		"""
		
		if self.verbose: print "worker : started job %s" % job.uuid
		try:
			job.model.train()
		except Exception, e:
			job.exception = e
			print e
		
		if self.verbose: print "worker: start evaluating recommender for %s"%job.uuid
		try:
			job.evaluator.model = job.model
			job.evaluator.newEval()
		except Exception, e:
			job.exception = e
			print e
		self.resultQueue.put(job)
		if self.verbose: print "worker : finish evaluating recommender for %s"%job.uuid	
		
class CrossValidation(object):
	"""
	Cette classe réalise une crossvalidation.
	"""
	
	
	def __init__(self, model, dataset, number):
		self.model = model
		self.dataset = dataset
		self.folds = Folds(dataset, number) # Todo: faire appel à la méthode qui génére des folds
		self.result = ThreadSafeContainer()
		self.threadNbSem = multiprocessing.Semaphore(10)
		
		
		self.mf = ModelFactory()
		self.mf.threadNbSem = self.threadNbSem
	
	def run(self):
		
		threads_list = []
		self.mf.start()
		for i in self.folds:
			t = threading.Thread(target=self._worker, args=(i, i))
			threads_list.append(t)
			
		for t in threads_list:
			t.start()
			t.join()

	
	def _worker(self, train, test):
		ticket = self.mf.add(self.model, train)
		model = self.mf.get(ticket)
		self.threadNbSem.acquire()
		process = multiprocessing.Process(target= self._performance, args= (model, test))
		process.start() 
		process.join()
	
	def _performance(self, model, test):
		self.result.append( model.predict(test) )
		#calcul de la performance
		self.threadNbSem.release()
		
		
		

if __name__ == "__main__":
	dataset = DataSet()
	dataset.matrix = numpy.array([i for i in range(120) ])
	dataset.matrix = dataset.matrix.reshape(12,10) 
	cs = CrossValidation(DummyLearner(), dataset, 12 )
	cs.run()

	print cs.result
	print "haha"
	
	
			
			
			
			
			
	

		
		
	
		
		


