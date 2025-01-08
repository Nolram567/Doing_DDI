#Installiere und lade die benötigten Pakete
install.packages("polmineR")
install.packages("dplyr")
install.packages("tidyr")
install.packages("jsonlite")
install.packages("devtools")

library(polmineR)
library(dplyr)
library(tidyr)
library(jsonlite)
library(devtools)

# Lade die Development-Version, da cwbtools derzeit im cran nicht verfügbar ist
devtools::install_github("PolMine/cwbtools", ref = "dev", force = TRUE) 

library(cwbtools)

# Das Korpus muss im Registry-Verzeichnis liegen und die Umgebungsvariable muss definiert sein.
gparl <- corpus("GERMAPARL2")

# Partition für die 17. bis 20. Legislaturperiode erstellen.
p <- partition(gparl, protocol_lp = c("17", "18", "19", "20"))

# KWIC-Abfrage durchführen
kwic_results <- kwic(p,
                     query="Dateninstitut",
                     left = 100,
                     right = 100,
                     s_attributes = c("protocol_date", "speaker_who",  "speaker_party", "protocol_url")
                     )

# Ergebnisse anzeigen
show(kwic_results)