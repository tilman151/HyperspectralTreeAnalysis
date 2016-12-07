# Protocol: 2016-12-07 (Framework-Tipps)

## Projektplan
- Corny hat Projektplan vorgestellt
- Zeitplan
- Auflistung aller Verfahren und Klassifikatoren

##Klassifikatoren mit Nachbarschaftsinformation
-SpaRSSE benötigt räumliche Informationen
-Deep Learning benötigt 5x5-Pixelfenster, braucht Koordinaten der gelabelten Daten

##Feature Extraction mit Nachbarschaftsinformation
- passt alles

##Datenverarbeitung
- Daten sind zu groß um sie gleichzeitig zu verarbeiten
- fürs Netzwerk ist das ok
- Möglichkeit: online SVMS/klassifikatoren
- Möglichkeit: Bilder zusammensetzen und "-1"-Rand lassen
---->Bilder zusammenfügen. Klassen mit nur einem Bild teilen = kleinste Fenstergröße

#Framework
- "-1" geschwärzt - Features auf Null
- -"0" ungelabelt - besitzen Featuredaten
- einige Bilder enthalten schon 0-Features
- Feature Extraction wird vom framework gemacht(aus config aufrufen)
- statische Feature Extraction

## vorläufige Ergebnisse
- t-SVM wirkt instabil (siehe results-Ordner)

## ToDo's:
- [ ] GIMP-Skill-Show (Tilman)
- [ ] Rotation Forest implementieren (Viola)
- [ ] PCA implementieren (Marianne)
- [ ] PCA auf random sample (MaTu)
- [ ] komplexeren Classifier implementieren (Corny)
- [ ] Random seed in Framework (Tilman)
- [ ] tex-Generierung aus matlab-Dateien (Tilman)
- [ ] function handles(Tilman)
- [ ] für feature extraction DAten rausziehen
- [ ] Projektplan abschicken
## zu klären
- Crossvalidierung -> Testsets sollten dieselbe Größe haben

###
14.12. ein Standardansatz plus continuum removal

