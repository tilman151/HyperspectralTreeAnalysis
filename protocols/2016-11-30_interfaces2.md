# Protocol: 2016-11-24 (Framework-Tipps)

## Config
- ein struct pro Algorithmus
- ein Parameter des structs ist create-handle
- Konstruktor erwartet Parameter einzeln, nicht das struct (Wiederverwendbarkeit)
- config als .m-File?

## Feature Extraction
- Christoph fragen, ob es methodisch in Ordnung ist, Feature Extraction auf allen Daten im Voraus zu machen
- Feature Extraction speichert und lädt nicht verarbeiteten Datensatz, sondern Transformationsmatrix
- Anschließende Anwendung der Transformation in der Feature Extraction
- Ausnahme: Indizes (werden jedes Mal neu berechnet, da wenig Aufwand), 
  Continuum Removal (je nach Zeit neu berechnen oder Datensatz abspeichern und laden)
  
## Partitionierung
- Interface für Klassifikatoren bleibt gleich, Framework "schwärzt" Testlabels
- Testinstanzen werden dann zu ungelabelten Daten -> Christoph fragen, ob methodisch in Ordnung

## Daten
- bei Uwe nachhaken, ob wir einen größeren Bildausschnitt mit mehreren Klassen bekommen

## vorläufige Ergebnisse
- t-SVM wirkt instabil (siehe results-Ordner)

ToDo's:
[ ] Fragen an Christoph (Marianne)
[ ] Uwe nach Daten fragen (Marianne)
[ ] Zeitplanentwurf (Corny)
[ ] Rotation Forest implementieren (Viola)
[ ] PCA implementieren (Marianne)
[ ] komplexeren Classifier implementieren (Corny)
[ ] Ensembles implementieren (Tuan)
[ ] Random seed in Framework (Tilman)
[ ] Umbau Partionierung (Tilman)
