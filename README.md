# ZIP-Bestanden

Tijdlab.ZIP (1.8GB):
- QGIS-Bestand
- Beroepsproduct
- Outputs
- Trainingsmodel
- Trainingskaarten
- Detectiekaarten

Data_Kaarten.zip (2.1GB):
- Originele kaarten zonder aanpassingen
- Data bodem- en geologische kaarten
- Handmatige Ground-Truth

# Van historische kaart naar digitale analyse

Automatische herkenning van cultuurhistorische landschapselementen met GIS en Python.

Dit project bevat een Python-workflow voor het automatisch herkennen van cultuurhistorische elementen op gegeorefereerde kadastrale minuutplannen uit 1832 rond Deventer. De workflow is ontwikkeld als proof-of-concept voor het herkennen van wegen en paden, watergangen en perceelgrenzen. De resultaten kunnen vervolgens in GIS worden gecontroleerd en gekoppeld aan bodemkundige en geologische gegevens.

## Doel van het project

Het doel van dit project is om historische kaartinformatie beter bruikbaar te maken voor archeologische en landschappelijke analyse. Kadastrale minuutplannen bevatten veel informatie over historische wegen, watergangen, percelen en landschapsstructuren, maar het handmatig digitaliseren daarvan is tijdrovend. Deze workflow onderzoekt of een deel van dat werk semi-automatisch kan worden ondersteund.

De methode is bedoeld als hulpmiddel bij:

- het herkennen van wegen, paden, watergangen en perceelgrenzen;
- het omzetten van historische kaartinformatie naar GIS-data;
- het controleren van automatische detectieresultaten;
- het vergelijken van historische structuren met bodemkundige en geologische gegevens;
- het ondersteunen van archeologisch bureauonderzoek en landschapsanalyse.

## Onderzoekskarakter

De workflow is opgezet als een reproduceerbaar proof-of-concept. De resultaten zijn nog niet bedoeld als volledig automatische of definitieve interpretatie. Menselijke controle blijft noodzakelijk, vooral bij kaartbladen met ruis, lage contrasten, tekst, legenda-elementen of onduidelijke perceelgrenzen.

## Hoofdlijnen van de workflow

De workflow bestaat uit de volgende stappen:

1. Inlezen van gegeorefereerde kadastrale minuutplannen in GeoTIFF-formaat.
2. Inlezen van handmatig geïnterpreteerde ground-truthlagen in GeoPackage-formaat.
3. Voorbewerking van de rasterbeelden.
4. Extractie van beeldkenmerken per pixel.
5. Training of hergebruik van een Random Forest-model.
6. Pixelgebaseerde classificatie van achtergrond, wegen/paden, watergangen en perceelgrenzen.
7. Morfologische opschoning van de voorspellingen.
8. Vectorisatie van de detectieresultaten.
9. Validatie met precision, recall, F1-score en confusion matrix.
10. Spatial join met bodemkundige en geologische gegevens.
11. Export van rasters, vectorbestanden, foutkaarten, metriekbestanden en frequentietabellen.

## Herkende klassen

| Klasse-ID | Klasse | Betekenis |
|---|---|---|
| 0 | Achtergrond | Niet tot een lijnobject behorende pixels |
| 1 | Wegen en paden | Historische wegen, paden en vergelijkbare lijnstructuren |
| 2 | Watergangen | Sloten, waterlopen en andere waterstructuren |
| 3 | Perceelgrenzen | Grenzen tussen historische percelen |

## Benodigde software

Voor de workflow wordt een aparte Python-omgeving aanbevolen. In het verantwoordingsdocument is hiervoor Miniforge met Python 3.11 gebruikt.

Benodigde pakketten:

- numpy
- pandas
- joblib
- rasterio
- geopandas
- shapely
- scikit-image
- scikit-learn
- opencv

## Installatie

Maak een nieuwe omgeving aan:

```bash
conda create -n tijdlab python=3.11
conda activate tijdlab
```

Installeer de benodigde pakketten via conda-forge:

```bash
conda install -c conda-forge numpy pandas joblib rasterio geopandas shapely scikit-image scikit-learn opencv
```

Controleer daarna of de belangrijkste pakketten werken:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import geopandas; print(geopandas.__version__)"
python -c "import rasterio; print(rasterio.__version__)"
```

## Bestanden voorbereiden

Controleer bovenaan `Tijdlab.py` of de vaste paden overeenkomen met de mappen op je eigen computer.

De workflow verwacht minimaal:

- gegeorefereerde trainingskaarten als `.tif` of `.tiff`;
- detectiekaarten als `.tif` of `.tiff`;
- een GeoPackage met handmatig geïnterpreteerde referentielagen;
- bodemkundige en geologische vectorlagen;
- een uitvoermap voor de resultaten.

De ground-truth moet minimaal de volgende lagen bevatten:

- `Wegen`
- `Water`
- `percelen_fixed`

Alle rasterkaarten moeten een geldige georeferentie en CRS bevatten. Kaarten zonder georeferentie worden door het script overgeslagen.

## Script uitvoeren

Activeer eerst de omgeving:

```bash
conda activate tijdlab
```

Ga daarna naar de map waarin `Tijdlab.py` staat:

```bash
cd pad/naar/de/projectmap
```

Voer het script uit:

```bash
python Tijdlab.py
```

## Model opnieuw trainen of hergebruiken

Onderaan het script bepaalt de functie `main()` of het model opnieuw wordt getraind of dat een bestaand model wordt gebruikt.

Nieuw model trainen:

```python
main(hergebruik_model=False)
```

Bestaand model hergebruiken:

```python
main(hergebruik_model=True)
```

Gebruik `False` wanneer de trainingsdata opnieuw moeten worden gebruikt om het model te trainen. Gebruik `True` wanneer het opgeslagen modelbestand `rf_model.joblib` al beschikbaar is.

## Belangrijkste uitvoer

Per kaartblad kan de workflow verschillende uitvoerbestanden maken.

| Bestand | Type | Doel |
|---|---|---|
| `ground_truth_mask.tif` | Raster | Controle van de gerasterde referentielagen |
| `prediction_classes.tif` / `classes.tif` | Raster | Voorspelde klasse per pixel |
| `mask_wegen.tif` | Raster | Deelmasker voor wegen en paden |
| `mask_water.tif` | Raster | Deelmasker voor watergangen |
| `mask_percelen.tif` | Raster | Deelmasker voor perceelgrenzen |
| `errormap.tif` | Raster | Foutkaart met correcte voorspellingen, false positives en false negatives |
| `metrics.csv` | Tabel | Precision, recall, F1-score en support per klasse |
| `confusion_matrix.txt` | Tekst/tabel | Verwarringsmatrix tussen echte en voorspelde klassen |
| `validatierapport.txt` | Tekstbestand | Samenvatting van de validatieresultaten per kaart |
| GeoPackage-output | Vectorbestand | Gedetecteerde wegen, watergangen en perceelgrenzen |
| Frequentietabellen | Tabel | Koppeling tussen gedetecteerde structuren en bodemkundige/geologische categorieën |

## Validatie

De kwaliteit van de detectie wordt beoordeeld door de automatische voorspellingen te vergelijken met handmatig geïnterpreteerde ground-truthlagen. Daarbij worden per klasse de volgende metrieken berekend:

- precision;
- recall;
- F1-score;
- support;
- confusion matrix.

Deze scores geven inzicht in de prestaties per objectklasse. Een hoge algemene score kan misleidend zijn, omdat achtergrondpixels veel talrijker zijn dan wegen, watergangen en perceelgrenzen. Daarom is vooral de prestatie per klasse belangrijk.

## Koppeling met bodem en geologie

Na de detectie worden de vectorresultaten via een spatial join gekoppeld aan bodemkundige en geologische gegevens. Hierdoor kan worden onderzocht in welke landschappelijke context de automatisch herkende structuren voorkomen. De frequentietabellen geven bijvoorbeeld weer hoe vaak wegen, watergangen of perceelgrenzen overlappen met bepaalde bodemkundige of geologische categorieën.

Deze koppeling levert geen direct bewijs voor oorzaak-gevolgrelaties, maar biedt wel een verkennende basis voor landschappelijke en archeologische interpretatie.

## Beperkingen

De workflow heeft verschillende beperkingen:

- de methode is getest op een beperkte set kadastrale minuutplannen;
- de kwaliteit van de detectie verschilt per kaartblad;
- perceelgrenzen zijn moeilijker betrouwbaar te herkennen dan wegen en watergangen;
- tekst, legenda’s, vlekken, scanruis en kaartverkleuringen kunnen foutpositieven veroorzaken;
- de resultaten moeten altijd visueel en inhoudelijk worden gecontroleerd;
- de koppeling met bodem en geologie is indicatief en afhankelijk van schaalverschillen tussen datasets.

## Projectstatus

Dit project is ontwikkeld als bachelor-afstudeeronderzoek voor Archeologie aan Hogeschool Saxion, uitgevoerd voor Tijdlab. De huidige versie is bedoeld als reproduceerbare onderzoeksworkflow en als basis voor vervolgonderzoek.

## Auteur

Merwen Yekdal  
Bachelor Archeologie, Hogeschool Saxion  
Tijdlab  
Juli 2026

## Licentie en gebruik

Controleer vóór hergebruik van data en kaartmateriaal altijd de licentievoorwaarden van de oorspronkelijke databronnen. De code en workflow kunnen worden gebruikt als basis voor vervolgonderzoek, mits de bron en auteur correct worden vermeld.
