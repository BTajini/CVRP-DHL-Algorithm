# -*- encoding: utf-8 -*-


#!!!!!!!!!!!!STATUS = In Progress (70%) (code not yet fixed for implementation) (need more time for managing reading from dataset and finding a way to stabilize the graph)!!!!!!!!!!!!!!!!

#Capacitated Vehicle Routing Problem (CVRP) 
#---Author--- : Badr Tajini 
#---Campus--- : Paris (On-Campus)
#---Release--- : 17/10/2016 - V3.0

#library implemented in the project
#-----------------------------------------------------------------------------------------------------
import sys, random
from math import sqrt, cells
import time
import pandas as pd   # library for reading our excel file
#----------------------------------------------------------------------------------------------------
#must edit the path for the dataset
excel_file_path = ("/CVRP-DHL-Algorithm/Data/Matrice_Plan_Transport.xls")   #add xlsx for excel 2010

df1 = pd.read_excel(open(excel_file_path,'rb'), sheetname="Sheet1",header = 0,index_col = 0,convert_float = True) #read demand of client
df3 = pd.read_excel(open(excel_file_path,'rb'), sheetname="Sheet3",header = 0,index_col = 0,convert_float = True) #read agency with depot
df4 = pd.read_excel(open(excel_file_path,'rb'), sheetname="Sheet4",header = 0,index_col = 0,convert_float = True) #read localisation of cities to get distance


#print the column names
print df1.columns
print df3.columns
print df4.columns
#print dataset
print (df1)
print (df3)
print (df4)
#get the values for a given column sheet 3
values3 = df3['code_agence','nom_agence','hub','depart','arrivee','cp'].values
#get the values for a given column sheet 4
values4 = df4['NAD_3251','X','Y'].values
#get the values for a given column sheet 1
values1 = df1['Rousset 01','Sitra 02','Moulinois 03','Gefco 04-05','Danzas 06','Masoye Gardon 07','Gefco 08','Fubra 09','TCP 10','CITE Messagerie 11','Cransac 12','Danzas 13','NormaTrans 14	15','Grimaud 16','Grimaud 17','Grimaud 19','Trancausse Marseille','Danzas 21','Arcatime 22','Vaquier 23-87','24','Danzas 25','Danzas 26','Danzas 28','Arcatime 29','Baldaroux 30','Danzas 31','Dubois 32','Danzas 33','Danzas 34','Arcatime 35','MRCI 36','Danzas 37','Danzas 38','Danzas 39','Dupuy 40','SPTG 41','Danzas 42','Archer 43','Danzas 49','Danzas 45','Querci Messagerie 46','Sernam 47','Arcatime 50','Danzas 51','Gondrand 52','Arcatime 53','Danzas 54','Arcatime 56','Danzas 57','LSH 58','Danzas 59','Sotrapoise 60','Arcatime 61','Coupé 62','Danzas 63','BMV 64','Danzas 65','Messagerie du midi 66','Danzas 67','Danzas 68','Danzas 69','Lesire 70','BMV 71','Arcatime 72','BMV 73','Danzas 74','Danzas 75','Danzas 76','Danzas 94','Arcatime 79','Prevote 80','NTM 81','Sodetram 83','Guyon 84','Arcatime 85','Arctime 86','Danzas 88','BMV 89','Danzas 90'].values

#get a data frame with selected columns
FORMAT3 = ['code_agence','nom_agence','hub','depart','arrivee','cp']
FORMAT4 = ['NAD_3251','X','Y']
FORMAT1 = ['Rousset 01','Sitra 02','Moulinois 03','Gefco 04-05','Danzas 06','Masoye Gardon 07','Gefco 08','Fubra 09','TCP 10','CITE Messagerie 11','Cransac 12','Danzas 13','NormaTrans 14	15','Grimaud 16','Grimaud 17','Grimaud 19','Trancausse Marseille','Danzas 21','Arcatime 22','Vaquier 23-87','24','Danzas 25','Danzas 26','Danzas 28','Arcatime 29','Baldaroux 30','Danzas 31','Dubois 32','Danzas 33','Danzas 34','Arcatime 35','MRCI 36','Danzas 37','Danzas 38','Danzas 39','Dupuy 40','SPTG 41','Danzas 42','Archer 43','Danzas 49','Danzas 45','Querci Messagerie 46','Sernam 47','Arcatime 50','Danzas 51','Gondrand 52','Arcatime 53','Danzas 54','Arcatime 56','Danzas 57','LSH 58','Danzas 59','Sotrapoise 60','Arcatime 61','Coupé 62','Danzas 63','BMV 64','Danzas 65','Messagerie du midi 66','Danzas 67','Danzas 68','Danzas 69','Lesire 70','BMV 71','Arcatime 72','BMV 73','Danzas 74','Danzas 75','Danzas 76','Danzas 94','Arcatime 79','Prevote 80','NTM 81','Sodetram 83','Guyon 84','Arcatime 85','Arctime 86','Danzas 88','BMV 89','Danzas 90']
df_selected3 = df[FORMAT3]
df_selected4 = df[FORMAT4]
df_selected1 = df[FORMAT1]


num_cities = 91
route = []
demand = 2000
capacity = 1000


start_score = 0
max_cities_to_visit = int(cells( ((num_cities-1) / float(capacity)) ))
best_distances = []
#city with index TRUE is depot
depot = df3[df3['hub'] == TRUE & df3['depart'] == TRUE & df3['arrivee'] == TRUE] # read depot with condititon true for the 3 columns ['hub','depart','arrivee']  / df3 local variable for opening excel sheet


#function to get track route
def get_route():
	global route
	lst = [i for i in xrange(num_cities)]
	for i in xrange( int(cells(num_cities / float(capacity))) ):
		track_route = []
		t_length = len(lst)-1 if len(lst)-1 < int(capacity) else capacity
		for j in xrange(t_length):
			choice = random.choice(lst[1:])
			lst.remove(choice)
			track_route.append(choice)
		route.append(track_route)
	return route


#function to get distance given for (x,y)
def get_distance_matrix(coords):
	"""
	Returns distance matrix of a given (x,y) coords
	"""
	matrix = {}
	for i, (x1, y1) in enumerate(coords):
		for j, (x2, y2) in enumerate(coords):
			dx = x1 - x2
			dy = y1 - y2
			dist = sqrt(dx*dx + dy*dy)
			matrix[i, j] = dist
	return matrix

	
#print column of sheet 3 (lat and log of all cities)
print df3.columns

#function to get coordination of all cities from the dataset
def get_cities_coords(num_cities, x, y):
	"""
	Calculate random position of a city (x,y - coord)
	"""
	coords = []
	for i in range(num_cities):
		x = random.randint(0, x)
		y = random.randint(0, y)
		coords.append( (float(x), float(y)) )
	return coords
	
#function for evaluating length of route
def eval_func(vehicle):
	"""
	The evaluation function
	"""
	global cm
	return get_route_length(cm, vehicle)

	cm = []
	coords = []
	
#create properties for vehicle
class vehicle:
	score = 0
	depot = True # has always true
	# init for create vehicle
	def __init__(self, vehicle=None, depot=0):
		self.vehicle = vehicle or self._makevehicle()
		self.score = 0
		self.depot = depot
		self.split_vehicle = self.split_route_on_capacity_with_depot()
	#function create a vehicle
	def _makevehicle(self):
		"""
		Makes a vehicle from randomly selected alleles
		"""
		vehicle = [self.depot]
		lst = [i for i in xrange(1,num_cities)]
		for i in xrange(1,num_cities):
			choice = random.choice(lst)
			lst.remove(choice)
			vehicle.append(choice)
		return vehicle
	#function for length of a route for current selection
	def evaluate(self):
		"""
		Calculates length of a route for current vehicle
		"""
		self.score = self.get_route_length()
	#cross two vehicle to get the best vehicle for current delivery
	def crossover(self, other):
		"""
		Cross two vehicle and returns best vehicle for delivery
		"""
		left, right = self._pickpivots()
		p1 = vehicle()
		p2 = vehicle()

		c1 = [c for c in self.vehicle[1:] if c not in other.vehicle[left:right+1]]
		p1.vehicle = [self.depot] + c1[:left] + other.vehicle[left:right+1] + c1[left:]
		c2 = [c for c in other.vehicle[1:] if c not in self.vehicle[left:right+1]]
		p2.vehicle = [other.depot] + c2[:left] + self.vehicle[left:right+1] + c2[left:]
	
	#print '====== ', p1, p2		
		return p1, p2
		
	#swap two vehicle
	def swap(self):
		"""
		Swap two elements
		"""
		left, right = self._pickpivots()
		self.vehicle[left], self.vehicle[right] = self.vehicle[right], self.vehicle[left]
		
	#return left or right direction of vehicle
	def _pickpivots(self):
		"""
		Returns random left, right pivots
		"""
		left = random.randint(1, num_cities - 2)
		right = random.randint(left, num_cities - 1)
		return left, right
		
	#copy a vehicle
	def copy(self):
		twin = self.__class__(self.vehicle[:])
		twin.score = self.score
		return twin

	#function to split route depending on capacity each vehicle
	def split_route_on_capacity_with_depot(self):
		"""
		Split route of cities [1,2,3,4] to routes depending on capacity
		"""
		split_list = []
		total_split_listy = 0

		while total_split_listy < (num_cities-1):
			length = random.randint(1, max_cities_to_visit)

			if length + total_split_listy < num_cities:
				total_split_listy += length
				split_list.append(length)
		
		step = 0
		self.split_routes = []
		for i,city in enumerate(split_list):
			route = [self.vehicle[0]] + self.vehicle[1+step:split_list[i]+step+1]
			step += split_list[i]
			self.split_routes.append(route)

		return self.split_routes
		
	#function to get length of route 
	def get_route_length(self):
		"""
		Returns the total length of the route
		"""
		total = 0
		global cm
	
		for track_route in self.split_routes:
			for i in xrange(len(track_route)):
				j = (i + 1) % len(track_route)
				city_from = track_route[i]
				city_to = track_route[j]
				total += cm[city_from, city_to]

		return total

	def __repr__(self):
		return '<%s vehicle="%s" score=%s>' % (self.__class__.__name__, str(self.split_vehicle), self.score)

#class for our network
class Network_route:
	demand = 0
	def __init__(self, Clients=None, demand=3, limits=2,\
			 	newvehicle=1, crossover_tx=1,\
			 	swap_tx=0.1):
		self.demand = demand
		self.Clients = self._makeclients()
		self.limits = limits
		self.newvehicle = newvehicle
		self.crossover_tx = crossover_tx
		self.swap_tx = swap_tx
		self.Step = 0
		self.minscore = sys.maxint
		self.minvehicle = None
		
	#function for create clients
	def _makeclients(self):
		return [vehicle() for i in xrange(0, self.demand)]
		
	# function for running routing step with best vehicle depending on capacity
	def run(self):
		for i in xrange(1, self.limits + 1):
			print 'Step no: ' + str(i) + '\n'
			for j in range(0, self.demand):
				self.Clients[j].evaluate()
			
				currentscore = self.Clients[j].score
				if currentscore < self.minscore:
					
					self.minscore = currentscore
					self.minvehicle = self.Clients[j].copy()
			print 'Best vehicle: ', self.minvehicle, ' ', id(self.minvehicle)

		

			#print self.minvehicle.vehicle
			if i == 1:
				start_score = self.minvehicle.score

			

		# crossover clients to create better delivery depending on capacity
			if random.random() < self.crossover_tx:
				children = []
				
				newindividual = int(self.newvehicle * self.demand )
				for i in xrange(0, newindividual):
					# select best vehicle to crossover
					selected1 = self._selectrank()
					while True:
						selected2 = self._selectrank()
						if selected1 != selected2:
							break

					client1 = self.Clients[selected1]
					client2 = self.Clients[selected2]
					delivery1, delivery2 = client1.crossover(client2)
					delivery1.evaluate()
					delivery2.evaluate()
					

					set_delivery1, set_delivery2 = False, False
				
					if delivery1.score < self.Clients[0].score:
						self.Clients.pop(0)
						self.Clients.append(delivery1)
						#print self.Clients
						set_delivery1 = True

					if delivery2.score < self.Clients[1].score:
						self.Clients.pop(1)
						self.Clients.append(delivery2)
						#print self.Clients
						set_delivery1 = True

					if not set_delivery1 and not set_delivery2:
						if delivery2.score < self.Clients[0].score:
							self.Clients.pop(0)
							self.Clients.append(delivery2)

						if delivery1.score < self.Clients[1].score:
							self.Clients.pop(1)
							self.Clients.append(delivery1)

			# swap 
			if random.random() < self.swap_tx:
				selected = self._select()	# select some vehicle to swap
				self.Clients[selected].swap()

		#end loop
		for i in xrange(0, self.demand):
			self.Clients[i].evaluate()
			currentscore = self.Clients[i].score
			if currentscore < self.minscore:
				self.minscore = currentscore
				self.minvehicle = self.Clients[i].copy()
				#print 'set min 2 ', self.minvehicle, ' ', self.minvehicle.score, ' ', id(self.minvehicle)

		print '.................Result.................'
		print self.minvehicle
	#function for select clients with demands
	def _select(self):
		totalscore = 0
		for i in xrange(0, self.demand):
			totalscore += self.Clients[i].score

		randscore = random.random() * (self.demand - 1)
		addscore = 0
		selected = 0
		for i in xrange(0, self.demand):
			addscore += (1 - self.Clients[i].score / totalscore)
			if addscore >= randscore:
				selected = i
				break
		return selected
		
	#function for select best vehicle to crossover depending on capacity and demand of clients
	def _selectrank(self):
		return random.randint(0,self.demand-1)

dist_matrix = []

#main function for running program
def main_run():
	global cm, coords, num_cities

	# get cities coords
	num_cities = 91
	coords = get_cities_coords(num_cities)
	cm = get_distance_matrix(coords)

	NR = Network_route(demand=100, limits=5)
	print("Step no: %d" % gen_num)
	print("Cities no: %d" % num_cities)
	print("Best distance: %.2f km" % best_score)
	print("Start distance: %.2f km" % start_score)
	print("Vehicle capacity: %d" % capacity)
	print("Client demands: %d" % demand)
	NR.run()

#run function main_run()
if __name__ == '__main__':
	#main_run()
