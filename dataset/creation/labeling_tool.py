from scrape_web_documents import resolve_doi

from selenium import webdriver
from selenium.webdriver.common.by import By

from tkinter import *

import os
import json
import sys
import shutil

import html
import re
import ast

if __name__ == '__main__':
    """
    Our labeling tool to assist the labeling of the dataset
    """

    # load anchor rules
    with open('anchor_rules.json', 'r') as f:
        anchor_rules = json.load(f)
    with open('anchor_whitelist.json', 'r') as f:
        anchor_whitelist = json.load(f)
    # load website rules
    with open('website_rules.json', 'r') as f:
        website_rules = json.load(f)
    with open('website_whitelist.json', 'r') as f:
        website_whitelist = json.load(f)
    with open('html_rules.json', 'r') as f:
        html_rules = json.load(f)

    # this tool is a helper for manual labeling of the dataset
    with open('dataset.json', 'r') as file:
        data = json.load(file)
    # data = [{'publisher_doi': '10.1109', 'doi': '10.3390/S90503337', 'key': 'conf/date/WagnerB09', 'authors': [('Ilya Wagner', ['University of Michigan, Ann Arbor, MI', 'test']), ('Valeria Bertacco', ['University of Michigan, Ann Arbor, MI'])], 'data': {'title': 'Caspar: Hardware patching <sub>for<\\sub> multicore processors.', 'year': '2009', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ISCAS.2000.858840', 'key': 'conf/iscas/HasanH00', 'authors': [('Mohammed A. Hasan', ['Dept of Electr. & Comput. Eng., Minnesota Univ., Duluth, MN, USA']), ('Ali A. Hasan', ['University of Minnesota Twin Cities'])], 'data': {'title': 'MUSIC and pencil-based sinusoidal estimation methods using fourth order cumulants.', 'year': '2000', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ESEM.2019.8870186', 'key': 'conf/esem/ViannaFG19', 'authors': [('Alexandre Vianna', ['Centro de Inform&#x00E1;tica, CIn Universidade Federal de Pernambuco, Recife, Brazil']), ('Waldemar Ferreira', ['Centro de Inform&#x00E1;tica, CIn Universidade Federal de Pernambuco, Recife, Brazil']), ('Kiev Gama', ['Centro de Inform&#x00E1;tica, CIn Universidade Federal de Pernambuco, Recife, Brazil'])], 'data': {'title': 'An Exploratory Study of How Specialists Deal with Testing in Data Stream Processing Applications.', 'year': '2019', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/CSNDSP.2016.7573993', 'key': 'conf/csndsp/Abu-AlmaalieTGL16', 'authors': [('Zina Abu-Almaalie', ['Optical Communications Research Group, NCRLab, Faculty of Engineering and Environment, Northumbria University, Newcastle Upon Tyne, United Kingdom']), ('Xuan Tang', ['Fujian Institute of Research on the Structure of Matter, Chinese Academy of Sciences, Fujian, China']), ('Zabih Ghassemlooy', ['Optical Communications Research Group, NCRLab, Faculty of Engineering and Environment, Northumbria University, Newcastle Upon Tyne, United Kingdom']), ('It Ee Lee', ['Faculty of Engineering, Multimedia University, 63100 Cyberjaya, Malaysia']), ('Alaa A. S. Al-Rubaie', ['Reconstruction and Projects Directorate, Ministry of Higher Education and Scientific Research, Baghdad, Iraq'])], 'data': {'title': 'Iterative multiuser detection with physical layer network coding for multi-pair communications.', 'year': '2016', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/AIM46487.2021.9517497', 'key': 'conf/aimech/PerreBCV21', 'authors': [('Greet Van de Perre', ['Vrije Universiteit Brussel, Robotics and Multibody Mechanics Research Group, Belgium', 'imec, Belgium']), ('Albert De Beir', ['imec, Belgium', 'Vrije Universiteit Brussel, Robotics and Multibody Mechanics Research Group, Belgium']), ('Hoang-Long Cao', ['Vrije Universiteit Brussel, Robotics and Multibody Mechanics Research Group, Belgium', 'Flanders Make@VUB']), ('Bram Vanderborght', ['imec, Belgium', 'Vrije Universiteit Brussel, Robotics and Multibody Mechanics Research Group, Belgium'])], 'data': {'title': 'Designing the social robot Elvis: how to select an optimal joint configuration for effective gesturing.', 'year': '2021', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/IRI-05.2005.1506471', 'key': 'conf/iri/SmariWPKMD05', 'authors': [('Waleed W. Smari', ['Dept. of Electr. & Comput. Eng., Dayton Univ., OH, USA']), ('Kirk Weigand', []), ('Gina Petonito', []), ('Yoga Kantamani', []), ('Rajitha Madala', []), ('Sirisha Donepudi', [])], 'data': {'title': 'An integrated approach to collaborative decision making using computer-supported conflict management methodology.', 'year': '2005', 'publisher': 'IEEE Systems, Man, and Cybernetics Society'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ISIT.2012.6283739', 'key': 'conf/isit/TalSV12', 'authors': [('Ido Tal', ['University of California San Diego, La Jolla, CA 92093, USA']), ('Artyom Sharov', ['Technion, Haifa, 32000, Israel']), ('Alexander Vardy', ['University of California San Diego, La Jolla, CA 92093, USA'])], 'data': {'title': 'Constructing polar codes for non-binary alphabets and MACs.', 'year': '2012', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ISCAS.2015.7169322', 'key': 'conf/iscas/DuanCLZC15', 'authors': [('Yan Duan', ['Dept. of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50010']), ('Tao Chen 0006', ['Dept. of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50010']), ('Zhiqiang Liu', ['Dept. of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50010']), ('Xu Zhang', ['Dept. of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50010']), ('Degang Chen 0001', ['Dept. of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50010'])], 'data': {'title': 'High-constancy offset generator robust to CDAC nonlinearity for SEIR-based ADC BIST.', 'year': '2015', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ITSC.2011.6083062', 'key': 'conf/itsc/JiG11', 'authors': [('Yuxuan Ji', ['Urban Transport Systems Laboratory (LUTS), &#x00C9;cole Polytechnique F&#x00E9;d&#x00E9;rate de Lausanne (EPFL)']), ('Nikolas Geroliminis', ['Urban Transport Systems Laboratory (LUTS), &#x00C9;cole Polytechnique F&#x00E9;d&#x00E9;rate de Lausanne (EPFL)'])], 'data': {'title': 'Exploring spatial characteristics of urban transportation networks.', 'year': '2011', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1109', 'doi': '10.1109/ISKE.2008.4731111', 'key': 'conf/iske/WuZ0H08', 'authors': [('Jianzhai Wu', ['Dept. of Autom. Control, National Univ. of Defense Technol., Changsha, China']), ('Zongtan Zhou', ['Dept. of Autom. Control, National Univ. of Defense Technol., Changsha, China']), ('Li Zhou 0003', ['Dept. of Autom. Control, National Univ. of Defense Technol., Changsha, China']), ('Dewen Hu', ['Dept. of Autom. Control, National Univ. of Defense Technol., Changsha, China'])], 'data': {'title': 'Learning spatial prior with automatically labeled landmarks.', 'year': '2008', 'publisher': 'IEEE'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/978-3-642-41218-9_10', 'key': 'conf/rsfdgrc/NguyenP13', 'authors': [('Sinh Hoa Nguyen', ['Polish-Japanese Institute of Inf. Technology Warszawa Poland']), ('Thi Thu Hien Phung', ['University of Economic and Technical Industries Ha Noi Viet Nam'])], 'data': {'title': 'Efficient Algorithms for Attribute Reduction on Set-Valued Decision Tables.', 'year': '2013', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/BF00196726', 'key': 'journals/joc/BengioBDGQ91', 'authors': [('Samy Bengio', ['Université de Montréal, Département IRO, Montréal, Canada']), ('Gilles Brassard', ['Université de Montréal, Département IRO, Montréal, Canada']), ('Yvo Desmedt', ['University of Wisconsin-Milwaukee, Department of EE & CS, Milwaukee, USA']), ('Claude Goutier', ['Centre de calcul, Université de Montréal, Montréal, Canada']), ('Jean-Jacques Quisquater', ['Université de Louvain, Département de Génie électrique (FAI), Louvain-la-Neuve, Belgium'])], 'data': {'title': 'Secure Implementations of Identification Systems.', 'year': '1991', 'publisher': None}}, {'publisher_doi': '10.1007', 'doi': '10.1007/978-3-540-45226-3_96', 'key': 'conf/kes/BaxterH03', 'authors': [('Jeremy W. Baxter', ['QinetiQ Ltd Malvern Technology Centre, St Andrews Road, Malvern, Worcestershire, WR14 3PS UK']), ('Graham S. Horn', ['QinetiQ Ltd Malvern Technology Centre, St Andrews Road, Malvern, Worcestershire, WR14 3PS UK'])], 'data': {'title': 'A Multi-agent System for Executing Group Tasks.', 'year': '2003', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/978-3-540-69134-1_21', 'key': 'conf/sab/MouretD08', 'authors': [('Jean-Baptiste Mouret', ['Université Pierre et Marie Curie-Paris6,CNRS FRE2507 ISIR Paris']), ('Stéphane Doncieux', ['Université Pierre et Marie Curie-Paris6,CNRS FRE2507 ISIR Paris'])], 'data': {'title': "Incremental Evolution of Animats' Behaviors as a Multi-objective Optimization.", 'year': '2008', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/S13198-023-01887-3', 'key': 'journals/saem/AroraMAK23', 'authors': [('Rajat Arora', ['University of Delhi, Department of Operational Research, Delhi, India']), ('Rubina Mittal', ['University of Delhi, Keshav Mahavidyalaya, Delhi, India']), ('Anu G. Aggarwal 0001', ['University of Delhi, Department of Operational Research, Delhi, India']), ('P. K. Kapur 0001', ['Amity University, Amity Center for Interdisciplinary Research, Noida, India'])], 'data': {'title': 'Investigating the impact of effort slippages in software development project.', 'year': '2023', 'publisher': None}}, {'publisher_doi': '10.1007', 'doi': '10.1007/978-3-319-11200-8_80', 'key': 'conf/ectel/PedroMP14', 'authors': [('Neuza Pedro', ['University of Lisbon Institute of Education Portugal']), ('João Filipe Matos', ['University of Lisbon Institute of Education Portugal']), ('Ana Pedro', ['University of Lisbon Institute of Education Portugal'])], 'data': {'title': "Digital Technologies, Teachers' Competences, Students' Engagement and Future Classroom: ITEC Project.", 'year': '2014', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/978-3-319-26350-2_3', 'key': 'conf/ausai/BaaderBL15', 'authors': [('Franz Baader', ['TU Dresden Theoretical Computer Science Dresden Germany']), ('Stefan Borgwardt', ['TU Dresden Theoretical Computer Science Dresden Germany']), ('Marcel Lippmann', ['TU Dresden Theoretical Computer Science Dresden Germany'])], 'data': {'title': 'Temporal Conjunctive Queries in Expressive Description Logics with Transitive Roles.', 'year': '2015', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/0-387-23572-8_9', 'key': 'conf/ifip3/KamajaL03', 'authors': [('Ilkka Kamaja', ['University of Lapland Department Research Methodology Lapland Finland']), ('Juha Lindfors', ['University of Lapland Department Research Methodology Lapland Finland'])], 'data': {'title': 'A Model for Planning, Implementing and Evaluating Client-Centered IT Education.', 'year': '2003', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/3-540-61735-3_17', 'key': 'conf/alp/KennawayOV96', 'authors': [('Richard Kennaway', ['university of east anglia']), ('Vincent van Oostrom', ['NTT BRL']), ('Fer-Jan de Vries', ['hitachi'])], 'data': {'title': 'Meaningless Terms in Rewriting.', 'year': '1996', 'publisher': 'Springer'}}, {'publisher_doi': '10.1007', 'doi': '10.1007/S10922-012-9247-Z', 'key': 'journals/jnsm/Chiang12', 'authors': [('Mao-Lun Chiang', ['Chaoyang University of Technology Department of Information and Communication Engineering Taichung County Republic of China'])], 'data': {'title': 'Efficient Diagnosis Protocol to Enhance the Reliability of a Cloud Computing Environment.', 'year': '2012', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.ENVSOFT.2013.11.001', 'key': 'journals/envsoft/GebbertP14', 'authors': [('Sören Gebbert', ['Thünen Institute of Climate-Smart Agriculture, Bundesallee 50, D-38116 Braunschweig, Germany']), ('Edzer J. Pebesma', [])], 'data': {'title': 'TGRASS: A temporal GIS for field based environmental modeling.', 'year': '2014', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.NEUROIMAGE.2008.03.016', 'key': 'journals/neuroimage/LiljestromTPKNHLS08', 'authors': [('Mia Liljeström', ['Brain Research Unit, Low Temperature Laboratory, Helsinki University of Technology, P.O. Box 5100 FIN-02015 TKK, Espoo, Finland']), ('Antti Tarkiainen', ['Advanced Magnetic Imaging Centre, Helsinki University of Technology, Espoo, Finland']), ('Tiina Parviainen', ['Brain Research Unit, Low Temperature Laboratory, Helsinki University of Technology, P.O. Box 5100 FIN-02015 TKK, Espoo, Finland']), ('Jan Kujala', ['Brain Research Unit, Low Temperature Laboratory, Helsinki University of Technology, P.O. Box 5100 FIN-02015 TKK, Espoo, Finland']), ('Jussi Numminen', ['Brain Research Unit, Low Temperature Laboratory, Helsinki University of Technology, P.O. Box 5100 FIN-02015 TKK, Espoo, Finland']), ('Jaana Hiltunen', ['Advanced Magnetic Imaging Centre, Helsinki University of Technology, Espoo, Finland']), ('Matti Laine', ['Department of Psychology, Åbo Akademi University, Turku, Finland']), ('Riitta Salmelin', ['Brain Research Unit, Low Temperature Laboratory, Helsinki University of Technology, P.O. Box 5100 FIN-02015 TKK, Espoo, Finland'])], 'data': {'title': 'Perceiving and naming actions and objects.', 'year': '2008', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/0306-4379(75)90002-2', 'key': 'journals/is/BernsteinT75', 'authors': [('Philip A. Bernstein', ['Department of Computer Science, University of Toronto, Toronto, Canada M5S 1A7']), ('Dennis Tsichritzis', ['Department of Computer Science, University of Toronto, Toronto, Canada M5S 1A7'])], 'data': {'title': 'Allocating Storage in Hierarchical Data Bases Using Traces.', 'year': '1975', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/0141-9331(90)90064-3', 'key': 'journals/mam/Souter90', 'authors': [('John Souter', ['Software Engineering Department, BSI Quality Assurance, PO Box 375, Milton Keynes MK14 6LL, UK'])], 'data': {'title': 'The position of MODULA-2 among programming languages.', 'year': '1990', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.NEUCOM.2017.03.040', 'key': 'journals/ijon/JiangJXH17', 'authors': [('Cuixia Jiang', ['School of Management, Hefei University of Technology, Hefei 230009, Anhui, PR China']), ('Ming Jiang', ['School of Management, Hefei University of Technology, Hefei 230009, Anhui, PR China']), ('Qifa Xu', ['School of Management, Hefei University of Technology, Hefei 230009, Anhui, PR China', 'Key Laboratory of Process Optimization and Intelligent Decision-making, Ministry of Education, Hefei 230009, Anhui, PR China']), ('Xue Huang', ['Department of Statistics, Florida State University, Tallahassee 32304, USA'])], 'data': {'title': 'Expectile regression neural network model with applications.', 'year': '2017', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.CAMWA.2018.10.029', 'key': 'journals/cma/LiH19', 'authors': [('Futuan Li', ['School of Mathematical Science, Zhejiang University, Hangzhou, 310027, China']), ('Xianliang Hu', ['School of Mathematical Science, Zhejiang University, Hangzhou, 310027, China'])], 'data': {'title': 'A phase-field method for shape optimization of incompressible flows.', 'year': '2019', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.CAM.2020.113017', 'key': 'journals/jcam/Boglaev21', 'authors': [('Igor Boglaev', ['Institute of Fundamental Sciences, Massey University, Private Bag 11-222, Palmerston North, New Zealand'])], 'data': {'title': 'A parameter robust numerical method for a nonlinear system of singularly perturbed elliptic equations.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.RESS.2021.108200', 'key': 'journals/ress/SongC22', 'authors': [('Kai Song', ['School of Management and Economics, Beijing Institute of Technology, Beijing, China']), ('Lirong Cui', ['College of Quality and Standardization, Qingdao University, Qingdao, China'])], 'data': {'title': 'A common random effect induced bivariate gamma degradation process with application to remaining useful life prediction.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/J.NEUROIMAGE.2020.117508', 'key': 'journals/neuroimage/ChenCPNRGQDPCV21', 'authors': [('Oliver Y. Chén', ['Department of Engineering Science, University of Oxford, Oxford OX1 4AR, United Kingdom']), ('Hengyi Cao', ['Center for Psychiatric Neuroscience, Feinstein Institute for Medical Research, Hempstead 11030, NY, United States', 'Department of Psychology, Yale University, New Haven 06510, CT, United States', 'Division of Psychiatry Research, Zucker Hillside Hospital, Glen Oaks 11004, NY, United States']), ('Huy Phan', ['School of Electronic Engineering and Computer Science, Queen Mary University of London, London E1 4NS, United Kingdom']), ('Guy Nagels', ['Department of Neurology, Universitair Ziekenhuis Brussel, 1090 Jette, Belgium']), ('Jenna M. Reinen', ['IBM Watson Research Center, Yorktown Heights, NY 10598, United States']), ('Jiangtao Gou', ['Department of Mathematics and Statistics, Villanova University, PA 19085, United States']), ('Tianchen Qian', ['Department of Statistics, Harvard University, Cambridge 02138, MA, United States', 'Department of Statistics, University of California Irvine, Irvine 92697, CA, United States']), ('Junrui Di', ['Department of Biostatistics, Johns Hopkins University, Baltimore 21205, MD, United States']), ('John Prince', ['Department of Engineering Science, University of Oxford, Oxford OX1 4AR, United Kingdom']), ('Tyrone D. Cannon', ['Department of Psychiatry, Yale University, New Haven 06510, CT, United States', 'Department of Psychology, Yale University, New Haven 06510, CT, United States']), ('Maarten De Vos', ['Faculty of Engineering Science, KU Leuven, Leuven 3001, Belgium', 'Faculty of Medicine, KU Leuven, Leuven 3001, Belgium', 'KU Leuven Institute for Artificial Intelligence, Leuven B-3000, Belgium'])], 'data': {'title': 'Identifying neural signatures mediating behavioral symptoms and psychosis onset: High-dimensional whole brain functional mediation analysis.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.1016', 'doi': '10.1016/0020-0190(95)00201-4', 'key': 'journals/ipl/Zimand96', 'authors': [('Marius Zimand', ['Department of Computer Science, University of Rochester, Rochester, NY 14627, USA'])], 'data': {'title': 'A High-Low Kolmogorov Complexity Law Equivalent to the 0-1 Law.', 'year': '1996', 'publisher': None}}, {'publisher_doi': '10.1145', 'doi': '10.1145/3447545.3451901', 'key': 'conf/wosp/RussoSC21', 'authors': [('Gabriele Russo Russo', ['University of Rome Tor Vergata, Rome, Italy']), ('Antonio Schiazza', ['University of Rome Tor Vergata, Rome, Italy']), ('Valeria Cardellini', ['University of Rome Tor Vergata, Rome, Italy'])], 'data': {'title': 'Elastic Pulsar Functions for Distributed Stream Processing.', 'year': '2021', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/3152896.3152899', 'key': 'conf/conext/SonCCKK17', 'authors': [('Donghyun Son', ['Seoul National University']), ('Eunsang Cho 0001', ['Seoul National University']), ('Minji Choi', ['Seoul National University']), ('Kay Khine', ['Seoul National University']), ('Ted Taekyoung Kwon', ['Seoul National University'])], 'data': {'title': 'Effects of partial infrastructure on indoor positioning for emergency rescue evacuation support system.', 'year': '2017', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/1181216.1181292', 'key': 'conf/siguccs/Sattler06', 'authors': [('Nicole M. Sattler', ['University of California at Berkeley, Berkeley, CA'])], 'data': {'title': 'Computer lab reservations: improving how to manage multiple reservation life cycles.', 'year': '2006', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/3331076', 'key': 'conf/ideas/2019', 'authors': [('Bipin C. Desai', ['Concordia University']), ('Dimosthenis Anagnostopoulos', ['Harokopio University of Athens']), ('Yannis Manolopoulos', ['Open University of Cyprus']), ('Mara Nikolaidou', ['Harokopio University of Athens'])], 'data': {'title': 'Proceedings of the 23rd International Database Applications & Engineering Symposium, IDEAS 2019, Athens, Greece, June 10-12, 2019.', 'year': '2019', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/1276958.1277057', 'key': 'conf/gecco/LiBK07', 'authors': [('Rui Li 0083', ['University of California, Riverside, CA']), ('Bir Bhanu', ['University of California, Riverside, CA']), ('Krzysztof Krawiec', ['University of California, Riverside, CA'])], 'data': {'title': 'Hybrid coevolutionary algorithms vs. SVM algorithms.', 'year': '2007', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/1180495.1180503', 'key': 'conf/vrst/KotranzaQL06', 'authors': [('Aaron Kotranza', ['University of Florida']), ('John Quarles', ['University of Florida']), ('Benjamin Lok', ['University of Florida'])], 'data': {'title': 'Mixed reality: are two hands better than one?', 'year': '2006', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/366552.366560', 'key': 'journals/cacm/Malcolm63', 'authors': [('W. David Malcolm Jr.', ['Minneapolis-Honeywell Regulator Co., Wellesley Hills, MA'])], 'data': {'title': 'String distribution for the polyphase sort.', 'year': '1963', 'publisher': None}}, {'publisher_doi': '10.1145', 'doi': '10.1145/223587.223629', 'key': 'conf/sigmetrics/SundaramE95', 'authors': [('C. R. M. Sundaram', ['Department of Computational Science, University of Saskatchewan, Saskatoon, SK S7N 0W0']), ('Derek L. Eager', ['Department of Computational Science, University of Saskatchewan, Saskatoon, SK S7N 0W0'])], 'data': {'title': 'Future Applicability of Bus-Based Shared Memory Multiprocessors.', 'year': '1995', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/3291533.3291579', 'key': 'conf/pci/ChristodoulouVT18', 'authors': [('Kyriakos Christodoulou', ['University of Athens, Athens, Greece']), ('Maria Vayanou', ['University of Athens, Athens, Greece']), ('George Tsampounaris', ['Athena Research and Innovation Center, Athens, Greece']), ('Yannis E. Ioannidis', ['Athena Research and Innovation Center, Athens, Greece'])], 'data': {'title': 'MagicARTS: an interactive social journey in the art world.', 'year': '2018', 'publisher': 'ACM'}}, {'publisher_doi': '10.1145', 'doi': '10.1145/3382734.3405728', 'key': 'conf/podc/KawarabayashiKS20', 'authors': [('Ken-ichi Kawarabayashi', ['NII']), ('Seri Khoury', ['UC Berkeley']), ('Aaron Schild', ['University of Washington']), ('Gregory Schwartzman', ['JAIST'])], 'data': {'title': 'Brief Announcement: Improved Distributed Approximations for Maximum-Weight Independent Set.', 'year': '2020', 'publisher': 'ACM'}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2203.12906', 'key': 'journals/corr/abs-2203-12906', 'authors': [('Anssi Moisio', []), ('Dejan Porjazovski', []), ('Aku Rouhe', []), ('Yaroslav Getman', []), ('Anja Virkkunen', []), ('Tamás Grósz', []), ('Krister Lindén', []), ('Mikko Kurimo', [])], 'data': {'title': 'Lahjoita puhetta - a large-scale corpus of spoken Finnish with some benchmarks.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2203.07902', 'key': 'journals/corr/abs-2203-07902', 'authors': [('Antoine Brochard', ['Dynamics of Geometric Networks']), ('Sixin Zhang', ['Algorithmes Parallèles et Optimisation']), ('Stéphane Mallat', ['École normale supérieure - Paris'])], 'data': {'title': 'Generalized Rectifier Wavelet Covariance Models For Texture Synthesis.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2205.00804', 'key': 'journals/corr/abs-2205-00804', 'authors': [('Marvin Zammit', ['University of Malta']), ('Antonios Liapis', ['University of Malta']), ('Georgios N. Yannakakis', ['University of Malta'])], 'data': {'title': 'Seeding Diversity into AI Art.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2206.06147', 'key': 'journals/corr/abs-2206-06147', 'authors': [('Adrien Cassagne', ['Architecture et Logiciels pour Systèmes Embarqués sur Puce']), ('Romain Tajan', ["Laboratoire de l'intégration, du matériau au système"]), ('Olivier Aumage', ['STatic Optimizations, Runtime Methods']), ('Camille Leroux', ["Laboratoire de l'intégration, du matériau au système"]), ('Denis Barthou', ['STatic Optimizations, Runtime Methods']), ('Christophe Jégo', ["Laboratoire de l'intégration, du matériau au système"])], 'data': {'title': 'A DSEL for High Throughput and Low Latency Software-Defined Radio on Multicore CPUs.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2206.03248', 'key': 'journals/corr/abs-2206-03248', 'authors': [('Aparup Khatua', ['L3S Research Center, Leibniz University Hannover']), ('Wolfgang Nejdl', ['L3S Research Center, Leibniz University Hannover'])], 'data': {'title': 'Rites de Passage: Elucidating Displacement to Emplacement of Refugees on Twitter.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.48550', 'doi': '10.48550/ARXIV.2206.06294', 'key': 'journals/corr/abs-2206-06294', 'authors': [('Valentin D. Richard', ['Semantic Analysis of Natural Language'])], 'data': {'title': 'Introducing Proof Tree Automata and Proof Tree Graphs.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S18082692', 'key': 'journals/sensors/ChenCLXWZ18', 'authors': [('Yujin Chen', ['State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China. yujin.chen@whu.edu.cn.']), ('Ruizhi Chen', ['Collaborative Innovation Center of Geospatial Technology (INNOGST), Wuhan 430079, China. ruizhi.chen@whu.edu.cn.']), ('Mengyun Liu', ['State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China. amylmy@whu.edu.cn.']), ('Aoran Xiao', ['State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China. xiaoaoran@whu.edu.cn.']), ('Dewen Wu', ['State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China. wudewen@whu.edu.cn.']), ('Shuheng Zhao', ['State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China. photonmango@foxmail.com.'])], 'data': {'title': 'Indoor Visual Positioning Aided by CNN-Based Image Retrieval: Training-Free, 3D Modeling-Free.', 'year': '2018', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S22082997', 'key': 'journals/sensors/ZouWLRLZBMZS22', 'authors': [('Xiuguo Zou', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Chenyang Wang 0005', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Manman Luo', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Qiaomu Ren', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Yingying Liu', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Shikai Zhang', ['College of Artificial Intelligence, Nanjing Agricultural University, Nanjing 210031, China']), ('Yungang Bai', ['College of Engineering, Nanjing Agricultural University, Nanjing 210031, China']), ('Jiawei Meng', ['Department of Mechanical Engineering, University College London, London WC1E 7JE, UK']), ('Wentian Zhang', ['Faculty of Engineering and Information Technology, University of Technology Sydney, Sydney, NSW 2007, Australia']), ('Steven W. Su', ['Faculty of Engineering and Information Technology, University of Technology Sydney, Sydney, NSW 2007, Australia'])], 'data': {'title': 'Design of Electronic Nose Detection System for Apple Quality Grading Based on Computational Fluid Dynamics Simulation and K-Nearest Neighbor Support Vector Machine.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S130709248', 'key': 'journals/sensors/MengYYX13', 'authors': [('Xianjing Meng', ['Shandong University(Shandong University),Jinan,China']), ('Yilong Yin', ['Shandong University(Shandong University),Jinan,China']), ('Gongping Yang', ['Shandong University(Shandong University),Jinan,China']), ('Xiaoming Xi', ['Shandong University(Shandong University),Jinan,China'])], 'data': {'title': 'Retinal Identification Based on an Improved Circular Gabor Filter and Scale Invariant Feature Transform.', 'year': '2013', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S21030780', 'key': 'journals/sensors/TakahashiM21', 'authors': [('Kazunori Takahashi', []), ('Takashi Miwa', ['Faculty of Science and Technology, Gunma University, Tenjin-cho 1-5-1, Kiryu, Gunma 376-8515, Japan.'])], 'data': {'title': 'A Local Oscillator Phase Compensation Technique for Ultra-Wideband Stepped-Frequency Continuous Wave Radar Based on a Low-Cost Software-Defined Radio.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S130809774', 'key': 'journals/sensors/ParkSCK13b', 'authors': [('Hyo Seon Park', ['Department of Architectural Engineering, Yonsei University, Seoul 110-732, Korea. hspark@yonsei.ac.kr']), ('Yunah Shin', []), ('Se Woon Choi', []), ('Yousok Kim', [])], 'data': {'title': 'Symbolic and Graphical Representation Scheme for Sensors Deployed in Large-Scale Structures.', 'year': '2013', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S21113715', 'key': 'journals/sensors/UngureanG21', 'authors': [('Ioan Ungurean', ['Faculty of Electrical Engineering and Computer Science, Stefan cel Mare University of Suceava, 720229 Suceava, Romania']), ('Nicoleta-Cristina Gaitan', ['Faculty of Electrical Engineering and Computer Science, Stefan cel Mare University of Suceava, 720229 Suceava, Romania'])], 'data': {'title': 'Software Architecture of a Fog Computing Node for Industrial Internet of Things.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/JIMAGING8020034', 'key': 'journals/jimaging/HiroseNTY22', 'authors': [('Ikumi Hirose', ['Division of Creative Engineering, Graduate School of Science and Engineering, Chiba University, Chiba 263-8522, Japan']), ('Kazuki Nagasawa', ['Division of Creative Engineering, Graduate School of Science and Engineering, Chiba University, Chiba 263-8522, Japan']), ('Norimichi Tsumura', ['Graduate School of Engineering, Chiba University, Chiba 263-8522, Japan']), ('Shoji Yamamoto', ['Tokyo Metropolitan College of Industrial Technology, Tokyo 140-0011, Japan'])], 'data': {'title': 'Texture Management for Glossy Objects Using Tone Mapping.', 'year': '2022', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S21020488', 'key': 'journals/sensors/SukhavasiSEAE21', 'authors': [('Susrutha Babu Sukhavasi', ['Department of Computer Science and Engineering, University Of Bridgeport, Bridgeport, CT 06604, USA']), ('Suparshya Babu Sukhavasi', ['Department of Computer Science and Engineering, University Of Bridgeport, Bridgeport, CT 06604, USA']), ('Khaled Elleithy', ['Department of Computer Science and Engineering, University Of Bridgeport, Bridgeport, CT 06604, USA']), ('Shakour Abuzneid', ['Department of Computer Science and Engineering, University Of Bridgeport, Bridgeport, CT 06604, USA']), ('Abdelrahman Elleithy', ['Department of Computer Science, William Paterson University, Wayne, NJ 07470, USA.'])], 'data': {'title': 'CMOS Image Sensors in Surveillance System Applications.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/E23111380', 'key': 'journals/entropy/HarounG21', 'authors': [('Mohamad F. Haroun', ['Department of Electrical and Computer Engineering, University of Victoria, P.O. Box\xa01700, STN CSC, Victoria, BC V8W 2Y2, Canada']), ('T. Aaron Gulliver', ['Department of Electrical and Computer Engineering, University of Victoria, P.O. Box\xa01700, STN CSC, Victoria, BC V8W 2Y2, Canada'])], 'data': {'title': 'Secure OFDM with Peak-to-Average Power Ratio Reduction Using the Spectral Phase of Chaotic Signals.', 'year': '2021', 'publisher': None}}, {'publisher_doi': '10.3390', 'doi': '10.3390/S20113108', 'key': 'journals/sensors/Abd-El-AttyIAE20', 'authors': [('Bassem Abd-El-Atty', ['Centre of Excellence in Cybersecurity, Quantum Information Processing, and Artificial Intelligence, Menoufia University, Shebin El-Koom 32511, Egypt.']), ('Abdullah M. Iliyasu', []), ('Haya Alaskar', []), ('Ahmed A. Abd El-Latif 0001', [])], 'data': {'title': 'A Robust Quasi-Quantum Walks-based Steganography Protocol for Secure Transmission of Images on Cloud-based E-healthcare Platforms.', 'year': '2020', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/BLTJ.20193', 'key': 'journals/bell/PetersonT07', 'authors': [('James S. Peterson', ['Mobility Architecture and Performance Department at Alcatel-Lucent in Naperville, Illinois|c|']), ('Joseph A. Tarallo', ['alcatel lucent'])], 'data': {'title': 'Wireless technology issue overview.', 'year': '2007', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/WCM.1048', 'key': 'journals/wicomm/WangWF12', 'authors': [('Qing Wang 0004', ['Department of Electronic Engineering, Tsinghua University, Beijing 100084, P.R. China']), ('Dapeng Wu 0001', ['Department of Electrical and Computer Engineering, University of Florida, Gainesville, FL 32611, U.S.A.']), ('Pingyi Fan', ['Department of Electronic Engineering, Tsinghua University, Beijing 100084, P.R. China and National Mobile Communications Research Laboratory, Southeast University, Nanjing 210096, China'])], 'data': {'title': 'Effective capacity of a correlated Nakagami-m fading channel.', 'year': '2012', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/JGT.3190170206', 'key': 'journals/jgt/Schrijver93', 'authors': [('Alexander Schrijver', ['CWI The Netherlands and Department of Mathematics University of Amsterdam, Amsterdam, The Netherlands'])], 'data': {'title': 'Note on hypergraphs and sphere orders.', 'year': '1993', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/CAE.10008', 'key': 'journals/caee/LopezS02', 'authors': [('Emilio A. Cariaga López', ['Departamento de Ciencias Matemáticas y Físicas, Universidad Católica de Temuco, Casilla 15‐D, Temuco, Chile']), ('Marcela C. Nualart Schindler', ['Escuela de Informática, Universidad Católica de Temuco, Casilla 15‐D, Temuco, Chile'])], 'data': {'title': 'Teaching and learning iterative methods for solving linear systems using symbolic and numeric software.', 'year': '2002', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/SPE.4380251103', 'key': 'journals/spe/Jaaksi95a', 'authors': [('Ari Jaaksi', ['Nokia Telecommunications, Network Management, Hatanpäänvaltatie 36 B, P.O. Box 779, 33100 Tampere, Finland'])], 'data': {'title': 'Object-oriented Specification of User Interfaces', 'year': '1995', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/JCC.23587', 'key': 'journals/jcc/XiaTYWHKL14', 'authors': [('Fei Xia', ['School of Biological Sciences, Nanyang Technological University; 60 Nanyang Drive 637551 Singapore']), ('Dudu Tong', ['School of Biological Sciences, Nanyang Technological University; 60 Nanyang Drive 637551 Singapore']), ('Lifeng Yang', ['School of Biological Sciences, Nanyang Technological University; 60 Nanyang Drive 637551 Singapore']), ('Dayong Wang', ['School of Computer Engineering, Nanyang Technological University; Nanyang Avenue 639798 Singapore']), ('Steven C. H. Hoi', ['School of Computer Engineering, Nanyang Technological University; Nanyang Avenue 639798 Singapore']), ('Patrice Koehl', ['Department of Computer Science and Genome Center; University of California; Davis California 95616']), ('Lanyuan Lu', ['School of Biological Sciences, Nanyang Technological University; 60 Nanyang Drive 637551 Singapore'])], 'data': {'title': 'Identifying essential pairwise interactions in elastic network model using the alpha shape theory.', 'year': '2014', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/NLA.1987', 'key': 'journals/nla/HezariES15', 'authors': [('Davod Hezari', ['Faculty of Mathematical Sciences; University of Guilan; Rasht Iran']), ('Vahid Edalatpour', ['Faculty of Mathematical Sciences; University of Guilan; Rasht Iran']), ('Davod Khojasteh Salkuyeh', ['Faculty of Mathematical Sciences; University of Guilan; Rasht Iran'])], 'data': {'title': 'Preconditioned GSOR iterative method for a class of complex symmetric system of linear equations.', 'year': '2015', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/RSA.10102', 'key': 'journals/rsa/AlonJMP03', 'authors': [('Noga Alon', ['Department of Mathematics, Sackler Faculty of Exact Sciences, Tel Aviv University, Tel Aviv, Israel']), ('Tao Jiang 0003', ['Department of Mathematics and Statistics, Miami University, Oxford, Ohio']), ('Zevi Miller', ['Department of Mathematics and Statistics, Miami University, Oxford, Ohio']), ('Dan Pritikin', ['Department of Mathematics and Statistics, Miami University, Oxford, Ohio'])], 'data': {'title': 'Properly colored subgraphs and rainbow subgraphs in edge-colorings with local constraints.', 'year': '2003', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/ROB.4620110503', 'key': 'journals/jfr/MatsunoY94', 'authors': [('Fumitoshi Matsuno', ['Department of Computer and Systems Engineering Faculty of Engineering Kobe University Rokkoudai, Nada Kobe 657, Japan']), ('Kazuo Yamamoto', ['Department of Computer and Systems Engineering Faculty of Engineering Kobe University Rokkoudai, Nada Kobe 657, Japan'])], 'data': {'title': 'Dynamic hybrid position/force control of a two degree-of-freedom flexible manipulator.', 'year': '1994', 'publisher': None}}, {'publisher_doi': '10.1002', 'doi': '10.1002/INT.4550050304', 'key': 'journals/ijis/Buckley90', 'authors': [('James J. Buckley', ['Mathematics Department, University of Alabama at Birmingham, Birmingham, AL'])], 'data': {'title': 'Belief updating in a fuzzy expert system.', 'year': '1990', 'publisher': None}}]
    checked_data = []
    labeled_websites_per_landingpage = {}
    blacklist = []
    save_location = 'checked_data.json'

    print('Loading data')
    # remove the data that has already been checked
    if os.path.exists(save_location):
        with open(save_location, 'r') as f:
            checked_data = json.load(f)

    # load labeled websites per landingpage if exists
    if os.path.exists('labeled_websites.json'):
        with open('labeled_websites.json', 'r') as f:
            labeled_websites_per_landingpage = json.load(f)

    # load blacklist if exists
    if os.path.exists('blacklist.json'):
        with open('blacklist.json', 'r') as f:
            blacklist = json.load(f)

    driver = webdriver.Firefox()


    def save_data():
        # copy the old save file to a backup
        if os.path.exists(save_location):
            shutil.copy(save_location, f'{save_location}.bak')
        # save the new data
        with open(save_location, 'w') as f:
            json.dump(checked_data, f)
        print('Saved metadata')

        # also safe the labeled websites
        if os.path.exists('labeled_websites.json'):
            shutil.copy('labeled_websites.json', 'labeled_websites.json.bak')
        with open('labeled_websites.json', 'w') as f:
            json.dump(labeled_websites_per_landingpage, f)

        # also safe the blacklist
        if os.path.exists('blacklist.json'):
            shutil.copy('blacklist.json', 'blacklist.json.bak')
        with open('blacklist.json', 'w') as f:
            json.dump(blacklist, f)


    for paper in data:

        print(paper)

        # check if the paper has already been checked (cointains 'labeled_websites' key)
        skip = False
        for x in checked_data:
            if x['doi'] == paper['doi']:
                # check if the key 'labeled_websites' exists
                if 'labeled_websites' in x:
                    print(f'{paper["doi"]} already checked')
                    skip = True
                    break
        if skip:
            continue

        landing_page = resolve_doi(paper['doi'])  # open the landing_page
        try:
            driver.get(landing_page)
        except Exception as e:
            print(f'Error opening {website}')
            print(e)

        # accept cookies if possible
        try:
            driver.find_element(By.XPATH,
                                '//*[contains(text(), "Accept") or contains(text(), "Save") or contains(text(), "Allow")]').click()
        except:
            pass

        # get all one_hop_websites
        doi_path = paper['doi'].replace('/', '\\')
        path_to_one_hop = os.path.join('websites', doi_path, 'one_hops')
        with open(os.path.join(path_to_one_hop, 'website.json'), 'r') as f:
            one_hop_websites = json.load(f)
        print(f'retrieved {len(one_hop_websites)} one_hop_websites')

        window = Tk()
        window.title("Label paper")
        window.geometry("800x600")

        entries = {}

        # Create a label and entry for publisher_doi
        label = Label(window, text='publisher_doi')
        label.pack()
        entry = Entry(window, width=250)
        insert = paper['publisher_doi'] if paper['publisher_doi'] else ''
        entry.insert(0, insert)
        entry.pack()
        entries['publisher_doi'] = entry

        # Create a label and entry for doi
        label = Label(window, text='doi')
        label.pack()
        entry = Entry(window, width=250)
        insert_text = paper['doi'] if paper['doi'] else ''
        entry.insert(0, insert_text)
        entry.pack()
        entries['doi'] = entry

        # Create a label and entry for title
        label = Label(window, text='title')
        label.pack()
        entry = Entry(window, width=250)
        insert_text = paper.get('data', {}).get('title', '') or ''
        entry.insert(0, insert_text)
        entry.pack()
        entries['title'] = entry

        # Create a label and entry for year
        label = Label(window, text='year')
        label.pack()
        entry = Entry(window, width=250)
        insert_text = paper.get('data', {}).get('year', '') or ''
        entry.insert(0, insert_text)
        entry.pack()
        entries['year'] = entry

        # Create a label and entry for publisher
        label = Label(window, text='publisher')
        label.pack()
        entry = Entry(window, width=250)
        insert_text = paper.get('data', {}).get('publisher', '') or ''
        print(insert_text)
        entry.insert(0, insert_text)
        entry.pack()
        entries['publisher'] = entry

        # create a label and text box for the authors and affiliations
        label = Label(window, text='authors and affiliations')
        label.pack()
        text = Text(window, width=250, height=10)
        author_text = paper['authors']
        if author_text:
            author_text_str = ""
            for author, affiliations in author_text:
                affiliations = [affil if affil else '' for affil in affiliations]
                affiliations_str = '", "'.join(affiliations)
                author_text_str += f'("{html.unescape(author)}", ["{html.unescape(affiliations_str)}"])\n'
            text.insert(1.0, author_text_str)
        else:
            author_schema = '("Author 1", ["Affiliation 1", "Affiliation 2"])\n("Author 2", ["Affiliation 1"])'
            text.insert(1.0, author_schema)
        text.pack()


        def save():
            result = {}
            result['publisher_doi'] = entries['publisher_doi'].get()
            result['doi'] = entries['doi'].get()
            result['data'] = {}
            result['data']['title'] = entries['title'].get()
            result['data']['year'] = entries['year'].get()
            result['data']['publisher'] = entries['publisher'].get()
            result['authors'] = []
            author_text = text.get(1.0, END)
            author_text = author_text.split('\n')
            for author in author_text:
                if author:
                    parsed_text = ast.literal_eval(author)

                    # Extract the author name and affiliations
                    author_name = parsed_text[0]
                    affiliations = parsed_text[1]
                    result['authors'].append((author_name, affiliations))
            print(result)

            # append one_hop_websites to checked_data
            result['one_hop_websites'] = one_hop_websites

            checked_data.append(result)

            save_data()

            window.destroy()


        button = Button(window, text="Save", command=save)
        button.pack()

        # some space between the buttons
        label = Label(window, text='')
        label.pack()


        def quit():
            save_data()

            window.destroy()
            driver.quit()
            # exit loop
            sys.exit()


        button = Button(window, text="Quit", command=quit)
        button.pack()

        window.mainloop()

        print(one_hop_websites)

        labeled_websites = {}
        # now for each one_hop_website, open the website and label with 1 or 0 whether it is relevant
        counter = 0
        for elem in one_hop_websites:
            anchor = elem['anchor']
            website = elem['website']

            found_rule = False

            # rulebased labeling
            for rule in anchor_rules:
                if re.match(rule, anchor):
                    print(f'{anchor} matches {rule}')
                    labeled_websites[website] = 0

                    found_rule = True
                    break
            if found_rule:
                save_data()
                continue

            for rule in website_rules:
                if re.match(rule, website):
                    print(f'{website} matches {rule}')
                    labeled_websites[website] = 0

                    found_rule = True
                    break
            if found_rule:
                save_data()
                continue

            for rule in anchor_whitelist:
                if re.match(rule, anchor):
                    print(f'{anchor} matches {rule}')
                    labeled_websites[website] = 1

                    found_rule = True
                    break
            if found_rule:
                save_data()
                continue

            for rule in website_whitelist:
                if re.match(rule, website):
                    print(f'{website} matches {rule}')
                    labeled_websites[website] = 1

                    found_rule = True
                    break
            if found_rule:
                save_data()
                continue

            for rule in html_rules:
                if rule == website:
                    print(f'{website} matches {rule}')
                    labeled_websites[website] = 0

                    found_rule = True
                    break
            if found_rule:
                save_data()
                continue

            # skip if already labeled
            if website in labeled_websites_per_landingpage.get(landing_page, {}):
                print(f'{website} already labeled')
                continue

            # if the website is in the blacklist, skip it
            if website in blacklist:
                print(f'{website} in blacklist')
                labeled_websites[website] = 0
                save_data()
                continue

            print(f'current website: {website}')
            try:
                driver.get(website)
            except Exception as e:
                print(f'Error opening {website}')
                print(e)

            window = Tk()
            window.title("Label website")
            window.geometry("800x800")

            print(paper['doi'])
            counter += 1

            # show the website url
            label = Label(window, text=website)
            label.pack()

            # show the known metadata (last in checked_data)
            if checked_data:
                last_checked = checked_data[-1]
                # Create a label and entry for publisher_doi
                label = Label(window, text='publisher_doi')
                label.pack()
                entry = Entry(window, width=250)
                insert_text = last_checked['publisher_doi'] if last_checked['publisher_doi'] else ''
                entry.insert(0, insert_text)
                entry.pack()

                # Create a label and entry for doi
                label = Label(window, text='doi')
                label.pack()
                entry = Entry(window, width=250)
                insert_text = last_checked['doi'] if last_checked['doi'] else ''
                entry.insert(0, insert_text)
                entry.pack()

                # Create a label and entry for title
                label = Label(window, text='title')
                label.pack()
                entry = Entry(window, width=250)
                insert_text = last_checked['data']['title'] if last_checked['data']['title'] else ''
                entry.insert(0, insert_text)
                entry.pack()

                # Create a label and entry for year
                label = Label(window, text='year')
                label.pack()
                entry = Entry(window, width=250)
                insert_text = last_checked['data']['year'] if last_checked['data']['year'] else ''
                entry.insert(0, insert_text)
                entry.pack()

                # Create a label and entry for publisher
                label = Label(window, text='publisher')
                label.pack()
                entry = Entry(window, width=250)
                insert_text = last_checked['data']['publisher'] if last_checked['data']['publisher'] else ''
                entry.insert(0, insert_text)
                entry.pack()

                # create a label and text box for the authors and affiliations
                label = Label(window, text='authors and affiliations')
                label.pack()
                text = Text(window, width=250, height=10)
                author_text = last_checked['authors']
                if author_text:
                    author_text_str = ""
                    for author, affiliations in author_text:
                        affiliations_str = '", "'.join(affiliations)
                        author_text_str += f'("{html.unescape(author)}", ["{html.unescape(affiliations_str)}"])\n'
                    text.insert(1.0, author_text_str)
                else:
                    author_schema = '("Author 1", ["Affiliation 1", "Affiliation 2"])\n("Author 2", ["Affiliation 1"])'
                    text.insert(1.0, author_schema)
                text.pack()


                # a big  button for 1 and a big button for 0
                def one():
                    labeled_websites[website] = 1
                    window.destroy()
                    print(f'Website {website} labeled as 1')


                button = Button(window, text="1", command=one, height=3, width=10)
                button.pack()

                # space between the buttons
                label = Label(window, text='')


                def zero():
                    labeled_websites[website] = 0
                    window.destroy()
                    print(f'Website {website} labeled as 0')


                button = Button(window, text="0", command=zero, height=3, width=10)
                button.pack()


                def blacklist_func():
                    blacklist.append(website)
                    window.destroy()
                    print(f'Website {website} added to blacklist')


                button = Button(window, text="Blacklist", command=blacklist_func, height=3, width=10)
                button.pack()


                # quit button
                def quit_label():
                    labeled_websites_per_landingpage[landing_page] = labeled_websites

                    save_data()

                    window.destroy()
                    driver.quit()
                    # exit loop
                    sys.exit()


                button = Button(window, text="Quit", command=quit_label, height=3, width=10)
                button.pack()

            window.mainloop()

        labeled_websites_per_landingpage[landing_page] = labeled_websites
        checked_data[-1]['labeled_websites'] = labeled_websites

        # save the labeled websites
        if os.path.exists('labeled_websites.json'):
            shutil.copy('labeled_websites.json', 'labeled_websites.json.bak')
        with open('labeled_websites.json', 'w') as f:
            json.dump(labeled_websites_per_landingpage, f)

    save_data()
