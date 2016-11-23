# Protocol: 2016-11-23 (Schnittstellen)

## Daten
- in HyperspectralTreeAnalysis/data/ftp-iff2.iff.fraunhofer.de

## Code conventions
- Parameter ohne Semikolon in der config
- Camelcase
- Dokumentation siehe Tuan
- externe Implementierungen in lib-Ordner

## Interfaces
- Cube + Labelmatrix als Eingabe für TrainOn
- Label als Ausgabe von Eval

## Implementierungen
- Feature Extraction
    - Indizes (Tilman)
    - Continuum Removal (läuft)
    - MC-LDA (läuft)
    - PCA (Marianne)
    - SELD (Marianne)
    - (gesplittete PCA - durch getrennte Daten nur begrenzt sinnvoll)
- Classification
    - SVM (Standard) (Corny)
    - Random Forest (Standard) (Viola)
    - t-SVM (Corny)
    - Rotation Forest (Viola)
    - (Convolutional Networks - aufwendig)
    - (Spatially Regularized Semi-Supervised Ensemble - aufwendig)
    - (Ensemble verschiedener Klassifikatoren)

## Sonstige Notizen
- Frage an Uwe: 
    - Wie auf mehreren Bildern mit nur einer Klasse trainieren?
    - Gesamtbild?
    - Einzelbilder skalieren und zusammenfügen eine Option?

## ToDos
- [ ] config überarbeiten (Kommentare, Validierung) - Tilman
- [ ] reset() als abstrakte Klasse des Classifier - Tilman
- [ ] Abstrakte Feature Extraction Klasse
- [ ] Check auf Existenz + Export der Daten in Feature Extraction
