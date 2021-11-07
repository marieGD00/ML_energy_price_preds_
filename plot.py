import matplotlib.pyplot as plt

#data_00b_season = {'All data': 0.9630097261117638,'Winter':0.9643476323556266,'Summer':0.9723785179403018,'Autumn':0.9449101567338972}
data_euclidean_season = {'All data':5.071961827374868,'Winter':6.056526696226041,'Summer':4.05424930814805,'Autumn': 4.476582184663146}
data_r2_season = {'All data': 0.9591836323646635,'Winter':0.9628166176001766,'Summer':0.9775144556181581,'Autumn':0.9283617597577163}


data_euclidean_momentjour= {'All data': 5.071961827374868,'00-6am':4.149907122836112,'6am-12pm':4.1494743532598495,'12pm-6pm':5.752533964283855,'6pm-12am':4.535093493249445}
data_r2_momentjour = {'All data': 0.9591836323646635,'00-6am':0.9273615182476317,'6am-12pm':0.968175662971458,'12pm-6pm':0.9655484633459535,'6pm-12am':0.9565612174399173}

#toutes donnees sans Zonal
data_r2_v2 = {'Sans Total': 0.9577715816103203,'Sans Zonal':0.9587624641603635}
data_euclidean_v2 = {'Sans Total': 5.1589486807063345,'Sans Zonal': 5.098062469156645}

data_prix_decalees_present = {'Avec prix decalees': 4.989500678775071,'Sans Prix decalees':5.857591270433541}
#toutes donnees sans Total

data_regression = {'All data': 0.7626848767196652 }
#data_regression = {sans}

#Gini Importance avec prix decalees
'''
Variable: prix decalees        Importance: 0.94
Variable: Unnamed: 0           Importance: 0.01
Variable: Heure                Importance: 0.01
Variable: Instant              Importance: 0.01
Variable: Posan                Importance: 0.01
Variable: Forecasted.Total.Load Importance: 0.01
Variable: Annee                Importance: 0.0
Variable: Mois                 Importance: 0.0
Variable: Jour                 Importance: 0.0
Variable: Minute               Importance: 0.0
Variable: JourSemaine          Importance: 0.0
Variable: JourFerie            Importance: 0.0
'''

#Gini Importance sans prix decalees
'''
Variable: Forecasted.Total.Load Importance: 0.5
Variable: Posan                Importance: 0.28
Variable: Annee                Importance: 0.1
Variable: Jour                 Importance: 0.06
Variable: JourSemaine          Importance: 0.02
Variable: Heure                Importance: 0.01
Variable: Instant              Importance: 0.01
Variable: Mois                 Importance: 0.0
Variable: Minute               Importance: 0.0
Variable: JourFerie            Importance: 0.0
'''


names = list(data_prix_decalees_present.keys())
values = list(data_prix_decalees_present.values())
plt.bar(names,values,color='g')
plt.ylim([4,6])
'''
plt.xlabel("type of data used")
plt.ylabel("R2 score")
plt.title(" R2 score")
'''

plt.xlabel("type of data used")
plt.ylabel("distance between estimated and real price ($)")
plt.title(" euclidean distance betweem predicted and true price")

'''
plt.xlabel("type of data used")
plt.ylabel("Euclidean distance btw predictions and true values")
plt.title(" Distance between prediction and true values, (feeding all variables)")
'''
'''
plt.xlabel("type of data used")
plt.ylabel("OOB score")
plt.title(" OOB estimate, (feeding all variables)")
'''
plt.show()