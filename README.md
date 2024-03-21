Our objective is to predicting survival state of patients with liver cirrhosis which is classification problem, the survival states include  D(death), C(censored), CL(censored due to liver transplantation). Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. Our data is inbalanced. We have 19 features and 1 target
1.Id: unique feature which we can't use
2.N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
3.Drug: type of drug D-penicillamine or placebo
4.Age: age
5.Sex: M(male) or F(female)
6.Ascites: it is a condition in which fluid collects in spaces within your abdomen. Presence of ascites is Y(yes) and oppesite is N(no)
7.Hepatomegaly: it is an enlarged liver, which means it's swollen beyond its usual size. Presence of hepatomegaly is Y(yes) and oppesite is N(no)
8.Spiders: it is a collection of blood vessels under the surface of your skin that resembles a spider. Presence of Spiders is Y(yes) and oppesite is N(no)
9.Edema: it is swelling caused by too much fluid trapped in the body's tissues. Presence of edema N(no edema and no diuretic therapy for edema), S(edema present without diuretics, or edema resolved by diuretics), or Y(edema despite diuretic therapy)
10.Bilirubin: it is is a red-orange compound that occurs in the normal catabolic pathway that breaks down heme in vertebrates. This catabolism is a necessary process in the body's clearance of waste products that arise from the destruction of aged or abnormal red blood cells. We have  the amount of it in blood. It is measured by mg/dl
11.Cholesterol: it is a waxy substance found in your blood. Your body needs cholesterol to build healthy cells, but high levels of cholesterol can increase your risk of heart disease
12.Albumin: it is a protein made by your liver. It enters your bloodstream and helps keep fluid from leaking out of your blood vessels into other tissues. It is measured by gm/dl
13.Copper: it is a soft red-brown metal. Your body needs small amounts of copper from food to stay healthy. But too much copper is toxic for liver. It is measured by ug/day
14.Alk_Phos: alkaline phosphatase (ALP) is a protein found in all body tissues. Tissues with higher amounts of ALP include the liver. It is measured by U/liter
15.SGOT: A glutamic-oxaloacetic transaminase (SGOT) or aspartate aminotransferase (AST) test measures the levels of the enzyme AST in the blood to assess liver health. It is measured by U/ml
16.Trylicerides: it is a blood test that measures the amount of a fat in your blood called triglycerides
17.Platelets: it is pieces of very large cells in the bone marrow called megakaryocytes. They help form blood clots to slow or stop bleeding and to help wounds heal The platelet count decreases with worsening liver disease. It is measured by ml/1000(platelets per cube)
18.Prothrombin: it is a protein made by the liver. It is one of several substances known as clotting (coagulation) factors. When you get a cut or other injury that causes bleeding, your clotting factors work together to form a blood clot. How fast your blood clots depends on the amount of clotting factors in your blood and whether they're working correctly. It is measured by s
19.Stage: histologic stage of disease(1,2,3 or 4)
