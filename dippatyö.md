# Otisskko tähän

## Tiivistelmä

Koneoppimis sovellutuksissa datan hankinta on usein yksi suurimpia haasteita. Tämä työ käsittelee, miten kahdella yleisemmin saatavilla olevasta datamallista, syvyysdata sekä segmentointi data voitaisiin hyödyntää rakentamaan mallia, joka pystyisi kuvasta tunnistamaan taustalla olevan ympäristön syvyyden, jättäen irtonaiset objektit huomiotta.

Datan käsittelyyn käytetään semanttista segmentointia, stereo kuvista haettua syvyyttä, sekä algoritmisesti syvyysdatasta objektien poistoa ja niiden takana olevien syvyyden arvioimista. Tästä johtuen lopputulos ei ole todennäköisesti kovin luotettava, mutta lopputuloksen pitäisi silti olla riittävä joihinkin ei kriittisiin applikaatioihin.

## Johdanto

Koneoppimisella voidaan hakea ratkaisua monenlaisiin ongelmiin. Yhtenä suurimpana esteenä sen toetuettamisessa on kuitenkin hyvän koulutusdatan hankkiminen. Erikoisemmissa tapauksissa, joissa datan käsittelyksi ei riitä vain kuvien eri alueiden luokittelu, onkin tästä syystä erittäin hankala löytää sopivia datamalleja. Tälläisissä tilanteissa myös totuuden määrittäminen hankalaoituu, koska datasta haetaan kuva-analyysillä tietoja joita edes ihminen ei voi kuvasta suoraan kertoa.

Kuvan käsittelyä ja siitä datan tunnistamista on tehty paljon ennen koneoppimisen yleistymistä. Näin ollen on luonnollista, että koneoppimis mallien koulutusdataa on parsittu sen avulla. Ja lopusksi saadun datan avulla voidaan esitellä kone-oppimis malli joka virtaviivaistaa tämän koko prosessin.

Tämä työn on tarkoitus tutkia, millaisia lopputuloksia on mahdollista saada rakentamalla, dataset stereodatan sekä semanttisen segmentoinnin pohjalta, josta algoritmillisesti poistetaan irtonaiset objektit ja niiden takana oleva syvyys arvioidaan. Lopputuloksena syntyvän mallin avulla, voidaan kuvan perusteella arvioida tilan tai alueen todellista statusta kun helposti liikutettavat asiat on sieltä poistettu tai siirrtyvät.

Tälläinen malli voisi pidemmälle jalostettuna, olla hyödyllinen monissa eri käyttötarkoituksissa. Esimerkiksi itseajavien laitteiden käytössä muuttuvissa ympäristöissä. Tai tilojen 3d scannauksessa viihde tai suunnittelu käyttöön. Pidemmälle jalostettuna tämänlaista mallia voisi hyödyntää esimerkiksi pelien tasojen 3d skannaukseen suoraan siirreltävillä tavaroilla varusteltuna.

Käytettävä data on itseajavien autojen kehitykseen liittyvää kaupunkidataa. Josta löytyy valmiina, syvyysdata. Kuitenkin koska mallin ja sen rakentamistyökalujen tulee olla hyödynnettävissä myös muulla datalla, käydään läpi myös stereokuvasta syvyysdatan hankinta. Segmentointi dataa ei uudelleen generoida, koska segmentointimalleja on paljon käytettävissä, jolloin sen itse rakentaminen tuskin on tarpeellista.


## Halutaanko kirjallisuus analyysi???

## Teoriaa 

### Neural networks

Mikä on neuroverkko

https://en.wikipedia.org/wiki/Artificial_neural_network



### stereo analysointi

Stereo kuvien syvyyttä voidaan arvioida ikkunan sovitus algoritmeillä, englanniksi block maching. Tämä tarkoittaa kahden sivusuunnassa eri kohdista otettujen kuvien sisällön vertaamista, siten että löydetään piste jossa sama asia esiintyy. [Kuva täältä](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)

Näin ollen jos tiedämme myös kameroiden etäisyyden toisistaan voimme trigonometiran abulla arvioida kuinka kaukana kyseinen kohde on. Tässä tapauksessa emme kuitenkaan ole kiinnostuneita siitä, vaan kohteiden suhteellinen etäisyys on meille riittävä tieto.

#### SGM Semi global matching

Selitä tähän miten semi global matching toimii

https://en.wikipedia.org/wiki/Semi-global_matching


### Semantic segmentation

[Selitä tähän miten semantic segmentation toimii](https://en.wikipedia.org/wiki/Image_segmentation)


### 3d point cloudit

[3d point ccloudeista vähän](https://en.wikipedia.org/wiki/Point_cloud)


## Käytettävä aineisto 

Datasettinä päädyttiin käyttämään [cityscapes](https://www.cityscapes-dataset.com/dataset-overview/#features) datasettiä. Kyseinen data pitää sisällään 3d sekä segmentation datan. Näitä datoja käsitelemällä voidaan saada aikaan haluttu lopputulos. Datasta oli tarjolla myös stereo kuva, jonka avulla voidaan tehdä oma sysvyyskartta, jos mukaan tarjotun kartan tarkkuus ei ole riittävä. SItä voi myös hydyntää oman toteutuksen toimivuuden vertailussa.

Aineisto on itseajaviin autojen toimintaan suunniteltu, joka tarkoittaa että lopullinen malli ei ole yleiskäyttöinen. Kuitenkin sin prosessointiin käytettäviä tekniikoita voidaan käyttää möys yleisen tai esimerkiksi sisätilojen skannaamiseen tarkoitetut mallin tekemiseen.

Aineistossa ei myöskään ole mallinnettu objektien takana olevaa syvyyttä, eikä sitä aineistosta voikkaan saada irti. Tästä johtuen ainestoa tulee hieman soveltaa koulutusdatan rakentamista varten.

## Metodilogia

Yhdistämällä kaksi tiedettyä ja manuaalisella valvonnalla voidaan yrittää toteuttaa järjestelmä joka pääsee haluttuun lopputulokseen


### Datasetin muodostus

#### Stereo analyysi

Aineistossa on jo generoitu syvyysdata. Tästä huolimatta olemme myös uudelleen generoineet sen jotta kehitetty koodi olisi paremmin hyöydynnettävissä myöhemmissä projekteissa.

#### Semantic segmentation

Aineistossa on jo totuus segmentaatio datasta. Käytämme tätä valmiina saatavaa dataa, mutta tämä koulutetaan silti uudelleen. Koulutettu verkko on hyvin yksinkertainen.

KERRO MEJÄN VERKOSTA


### Miten data yhdistettiin

Prosessointi koodi. 

Tunnista autot ja muut, depth mapista. 
Aseta gradient ympäröivein juttujen mukaan

### miten data validoitiin

JOnkinnäköinen käyttöliittymä datan validointiin ja generointiin

## Johtopäätökset

Oliko kerätty data riittävän hyvää??

### tekeekö datalla mitään 

### Kehitysehdotukset

### muut käyttökohteet

## Lähteet