import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.font_manager as ftman
import matplotlib.colors as colors
import matplotlib.cm as cmx


tatanic = pd.read_csv('train.csv')
#print titanic.head()
#print titanic.describe()
#print titanic.info()

figH, figW = (8.5, 11.0)
fig = plt.figure(figsize=(figW, figH), dpi=300)
fig.set(alpha=0.2)


# Statistics of servived people
plt.subplot2grid((2,3),(0,0))            
tatanic.Survived.value_counts().plot(kind='bar')
plt.title("Survived") 
plt.ylabel("Number of People")  


# Statistics of servived people in different Pclasses.
plt.subplot2grid((2,3),(0,1))
tatanic.Pclass.value_counts().plot(kind="bar")
plt.title("Survived vs Pclass")
plt.ylabel("Number of Pelple")
plt.xlabel("Pclass")


# Statistics of ages of the surivived and dead people 
plt.subplot2grid((2,3),(0,2))
plt.scatter(tatanic.Survived, tatanic.Age)
plt.grid(b=True, which='major', axis='y') 
plt.title("Distribution of Age")
plt.ylabel("Age")    


#  Distribution of people ages in different PClass
plt.subplot2grid((2,3),(1,0), colspan=2)
tatanic.Age[tatanic.Pclass == 1].plot(kind='kde')   
tatanic.Age[tatanic.Pclass == 2].plot(kind='kde')
tatanic.Age[tatanic.Pclass == 3].plot(kind='kde')
plt.title("Age Distribution of different Pclasses")
plt.xlabel("Age")
plt.ylabel("Density") 
plt.legend(('1st Class', '2nd Calss','3rd Class'),loc='best') 


plt.subplot2grid((2,3),(1,2))
tatanic.Embarked.value_counts().plot(kind='bar')
plt.title("Embarked")
plt.ylabel("Number of People")  

fig.savefig("Titanic_Overview.PDF")


###############################################################################

figH, figW = (11.0, 8.5)
fig = plt.figure(figsize=(figW, figH), dpi=300)
fig.set(alpha=0.2)


ax = plt.subplot2grid((3,2),(0,0))            
Survived_0 = tatanic.Pclass[tatanic.Survived == 0].value_counts()
Survived_1 = tatanic.Pclass[tatanic.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'Dead':Survived_0})
df.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'])
plt.title("Survived vs PCalss")
plt.xlabel("Plcass")
plt.ylabel("Number of People")


ax = plt.subplot2grid((3,2),(0,1))            
Survived_m = tatanic.Survived[tatanic.Sex == 'male'].value_counts()
Survived_f = tatanic.Survived[tatanic.Sex == 'female'].value_counts()
df=pd.DataFrame({'Male':Survived_m, 'Female':Survived_f})
df.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'])
plt.title("Survived vs Sex")
plt.xlabel("Sex")


ax = plt.subplot2grid((3,2),(1,0), colspan=2)
Survived_fh = tatanic.Survived[tatanic.Sex == 'female'][tatanic.Pclass != 3].value_counts()
Survived_fl = tatanic.Survived[tatanic.Sex == 'female'][tatanic.Pclass == 3].value_counts()
Survived_mh = tatanic.Survived[tatanic.Sex == 'male'][tatanic.Pclass != 3].value_counts()
Survived_ml = tatanic.Survived[tatanic.Sex == 'male'][tatanic.Pclass == 3].value_counts()
df = pd.DataFrame({"Female Highclass": Survived_fh, 'Female Lowclass': Survived_fl, 'Male Highclass': Survived_mh, 'Male Lowclass': Survived_ml})
df.plot(kind='bar', ax=ax, color=['magenta', 'pink', 'steelblue', 'lightblue'])
ax.set_xticklabels(["Dead", "Survived"], rotation=0)
ax.legend(loc='best')


ax = plt.subplot2grid((3,2), (2,0) )
Survived_0 = tatanic.Embarked[tatanic.Survived == 0].value_counts()
Survived_1 = tatanic.Embarked[tatanic.Survived == 1].value_counts()
df=pd.DataFrame({'Survived':Survived_1, 'Dead':Survived_0})
df.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'])
plt.xlabel("Embarked")
plt.ylabel("Number of People")


ax = plt.subplot2grid((3,2), (2,1) )
Survived_cabin = tatanic.Survived[pd.notnull(tatanic.Cabin)].value_counts()
Survived_nocabin = tatanic.Survived[pd.isnull(tatanic.Cabin)].value_counts()
df=pd.DataFrame({'Survived':Survived_cabin, 'Dead':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'])
ax.set_xticklabels(["Dead", "Survived"], rotation=0)
plt.xlabel("Cabin")
plt.show()

fig.savefig("Titanic_Overview_2.PDF")
