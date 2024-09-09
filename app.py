import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models
mode = st.sidebar.selectbox("Choose Input Mode", ["Plant Disease Classifier","Chatbot Mode", "Image to Text"])
if mode == "Plant Disease Classifier":
    INPUT_SIZE = 256
    MEAN_AUGMENTED = [0.4683, 0.5414, 0.4477]
    STD_AUGMENTED = [0.2327, 0.2407, 0.2521]
    import streamlit as st

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=300)

    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
        'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
        'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    translations = {
        'English': {
            'title': "Plant Disease Classifier",
            'upload_prompt': "Upload an image of a plant leaf to classify its disease.",
            'choose_image': "Choose an image...",
            'predicted_class': "Predicted Class",
            'confidence': "Confidence",
            'solution' : {'Apple___Apple_scab': (
                "Apple scab is a fungal disease. Apply fungicides containing captan or myclobutanil. "
                "Remove and destroy any fallen leaves or fruit to reduce the spread of the fungus. "
                "Ensure proper tree pruning to improve air circulation. Regular applications during the growing season are critical."
            ),
            'Apple___Black_rot': (
                "Black rot can be controlled by removing infected branches and fruit. Apply copper-based fungicides or liquid lime sulfur. "
                "Prune trees to improve air circulation and avoid overcrowding. During the growing season, fungicide sprays like captan or thiophanate-methyl are recommended."
            ),
            'Apple___Cedar_apple_rust': (
                "Cedar apple rust is best managed by applying fungicides containing myclobutanil or propiconazole at petal fall and repeating every 7-10 days. "
                "Remove any nearby cedar trees, as they are alternate hosts for the disease. Use resistant apple varieties when planting new trees."
            ),
            'Cherry_(including_sour)___Powdery_mildew': (
                "For powdery mildew, sulfur or potassium bicarbonate fungicides are effective. "
                "Ensure proper air circulation around the plant and avoid excessive nitrogen fertilization, which can promote soft, susceptible growth. "
                "Prune affected leaves and branches, and water at the base of the plant rather than from overhead."
            ),
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': (
                "Gray leaf spot can be managed by using fungicides like strobilurins (e.g., Azoxystrobin) or triazoles (e.g., Propiconazole). "
                "Remove crop residues and rotate crops to reduce pathogen carryover. Planting resistant hybrids is also recommended."
            ),
            'Corn_(maize)___Common_rust_': (
                "Common rust can be controlled with fungicides like mancozeb or chlorothalonil. "
                "Ensure proper crop rotation and avoid high plant densities to improve air circulation. "
                "Using rust-resistant hybrids is one of the most effective preventive measures."
            ),
            'Corn_(maize)___Northern_Leaf_Blight': (
                "Northern leaf blight can be controlled with fungicides like strobilurins (Azoxystrobin) and triazoles (Tebuconazole). "
                "Plant resistant hybrids and use crop rotation to break the disease cycle. "
                "Ensure fields are not over-irrigated, as moisture can increase the severity of the disease."
            ),
            'Grape___Black_rot': (
                "Black rot in grapes requires consistent fungicide application, including captan, myclobutanil, or mancozeb. "
                "Remove infected leaves and fruit promptly, and ensure proper pruning to increase air circulation. "
                "Begin fungicide treatment early in the growing season and continue throughout the season, especially during wet periods."
            ),
            'Grape___Esca_(Black_Measles)': (
                "Unfortunately, there is no cure for Black Measles. Focus on prevention by ensuring proper irrigation and avoiding injuries to the vine, which can let the fungus in. "
                "You may consider using a fungicide like thiophanate-methyl as a protective treatment, though this will not cure already infected plants. "
                "Remove and destroy severely infected vines to prevent the disease from spreading."
            ),
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': (
                "Leaf blight can be treated with fungicides like copper-based sprays or mancozeb. "
                "Ensure regular pruning to allow air circulation, and remove any infected leaves or fruit. "
                "Avoid overhead irrigation and opt for drip irrigation to minimize moisture on leaves."
            ),
            'Orange___Haunglongbing_(Citrus_greening)': (
                "Citrus greening is a serious disease with no known cure. Infected trees should be removed and destroyed to prevent spread. "
                "Control the Asian citrus psyllid, the insect vector of the disease, with insecticides like imidacloprid or thiamethoxam. "
                "Implement proper orchard management practices, including regular monitoring and removal of infected plants."
            ),
            'Peach___Bacterial_spot': (
                "Bacterial spot can be controlled using copper-based fungicides or oxytetracycline sprays. "
                "Remove and destroy infected leaves and fruit. Avoid overhead watering, as moisture encourages the spread of the bacteria. "
                "Choose resistant varieties when planting new peach trees, and prune trees to improve air circulation."
            ),
            'Pepper,_bell___Bacterial_spot': (
                "Control bacterial spot by applying copper-based bactericides regularly. "
                "Avoid overhead watering and ensure proper crop rotation. Remove infected leaves and fruits promptly to prevent further spread."
            ),
            'Potato___Early_blight': (
                "For early blight, apply fungicides like chlorothalonil or mancozeb. "
                "Rotate crops, avoid overhead watering, and ensure proper spacing between plants to reduce humidity. "
                "Remove and destroy infected plant debris to limit the spread of the disease."
            ),
            'Potato___Late_blight': (
                "Late blight can be managed using fungicides like copper-based products or systemic fungicides like mefenoxam. "
                "Destroy infected plants and tubers to prevent the spread. Avoid overhead watering and ensure proper soil drainage."
            ),
            'Squash___Powdery_mildew': (
                "For powdery mildew, apply fungicides like sulfur, neem oil, or potassium bicarbonate. "
                "Remove severely infected leaves, improve air circulation, and water plants at the base to avoid wetting the foliage. "
                "Avoid overcrowding plants, as this encourages the growth of mildew."
            ),
            'Strawberry___Leaf_scorch': (
                "Leaf scorch can be treated with fungicides like captan or copper-based sprays. "
                "Remove infected leaves and ensure proper spacing between plants to improve air circulation. Avoid overhead watering."
            ),
            'Tomato___Bacterial_spot': (
                "Bacterial spot can be controlled using copper-based bactericides or streptomycin. "
                "Remove infected leaves and fruit, and avoid working with wet plants to prevent spreading the bacteria."
            ),
            'Tomato___Early_blight': (
                "Early blight can be managed with fungicides like chlorothalonil, copper, or mancozeb. "
                "Prune lower leaves to improve air circulation, mulch to prevent soil from splashing on leaves, and remove infected leaves."
            ),
            'Tomato___Late_blight': (
                "Late blight can be treated using fungicides like copper-based products or mancozeb. "
                "Remove and destroy infected plants and avoid overhead irrigation. Use disease-resistant varieties where possible."
            ),
            'Tomato___Leaf_Mold': (
                "Leaf mold is controlled with fungicides like chlorothalonil or mancozeb. "
                "Ensure good air circulation, avoid overhead watering, and prune heavily infected leaves."
            ),
            'Tomato___Septoria_leaf_spot': (
                "Septoria leaf spot can be treated with fungicides like mancozeb or chlorothalonil. "
                "Prune affected areas and mulch plants to reduce the spread of spores from the soil."
            ),
            'Tomato___Spider_mites Two-spotted_spider_mite': (
                "Spider mites can be managed by increasing humidity and applying insecticidal soap or neem oil. "
                "Regularly spray the plant with water to dislodge mites, and consider introducing predatory mites to control the population."
            ),
            'Tomato___Target_Spot': (
                "Target spot can be treated with fungicides like chlorothalonil or copper-based products. "
                "Improve air circulation by proper spacing, prune infected areas, and avoid overhead irrigation."
            ),
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
                "There is no cure for the virus. Remove infected plants immediately to prevent spread. "
                "Control whiteflies, the insect vector, using insecticides like imidacloprid or neem oil. "
                "Use disease-resistant varieties where possible."
            ),
            'Tomato___Tomato_mosaic_virus': (
                "There is no cure for this virus. Remove infected plants and sanitize tools after working with infected plants. "
                "Avoid handling plants when wet, as the virus can easily spread through contact."
            ) }
        },
        'Tamil': {
            'title': "தாவர நோய் வகைப்படுத்தி",
            'upload_prompt': "உங்கள் செடி இலையின் படத்தை பதிவேற்றவும்.",
            'choose_image': "படத்தைத் தேர்ந்தெடுக்கவும்...",
            'predicted_class': "கணிக்கப்பட்ட வகை",
            'confidence': "நம்பிக்கை",
            'solution' : {'Apple___Apple_scab': (
        "ஆப்பிள் ஸ்காப் ஒரு பூஞ்சை நோயாகும். காப்டான் அல்லது மைக்லோபுடேநில் கொண்ட பூஞ்சிமருந்துகளை பயன்படுத்துங்கள். "
        "பழுதுபட்ட இலைகள் மற்றும் பழங்களை அகற்றி அழிக்கவும். பூஞ்சையின் பரவலைக் குறைக்கவும். "
        "மரத்தின் சரியான வெட்டுமுறையை உறுதி செய்யுங்கள், மேலும் காற்றோட்டத்தை மேம்படுத்தவும். வளர்ச்சிகாலத்தில் சீரான பயன்பாடுகள் மிகவும் முக்கியம்."
    ),
    'Apple___Black_rot': (
        "பிளாக் ராட் பாதிக்கப்பட்ட கிளைகளையும் பழங்களையும் அகற்றி கட்டுப்படுத்தலாம். காப்பர் அடிப்படையிலான பூஞ்சிமருந்துகள் அல்லது திரவ எலுமிச்சை சல்பரைப் பயன்படுத்துங்கள். "
        "மரங்களை வெட்டிக் காற்றோட்டத்தை மேம்படுத்துங்கள் மற்றும் இடைநிறுத்தம் தவிர்க்கவும். வளர்ச்சிகாலத்தில், காப்டான் அல்லது தியோஃபனேட்-மிதைல் போன்ற பூஞ்சிமருந்து தெளிப்புகள் பரிந்துரைக்கப்படுகின்றன."
    ),
    'Apple___Cedar_apple_rust': (
        "சீடர் ஆப்பிள் ரஸ்ட் மைக்லோபுடேநில் அல்லது ப்ரோபிகோனசோல் கொண்ட பூஞ்சிமருந்துகளை இதழ் விழுததில் பயன்படுத்தி, 7-10 நாட்களுக்கு ஒருமுறை மீண்டும் பயன்படுத்தி சிறந்த முறையில் கையாளப்படுகிறது. "
        "அருகில் உள்ள சீடர் மரங்களை அகற்றுங்கள், ஏனெனில் அவை நோய்க்கு மாற்று தோழர்கள் ஆகின்றன. புதிய மரங்களை நட்டபோது நோய்க்கு எதிர்ப்புடைய ஆப்பிள் வகைகளைப் பயன்படுத்துங்கள்."
    ),
    'Cherry_(including_sour)_Powdery_mildew': (
        "பவுட்ரி மில்டியூக்கு, சல்பர் அல்லது பொட்டாசியம் பைக்கார்பனேட் கொண்ட பூஞ்சிமருந்துகள் விளைவளிக்கின்றன. "
        "மரத்தைச் சுற்றி சரியான காற்றோட்டத்தை உறுதி செய்யுங்கள் மற்றும் மென்மையான, பாதிக்கக்கூடிய வளர்ச்சியை மேம்படுத்தும் அதிக நைட்ரஜன் உரத்தைத் தவிர்க்கவும். "
        "பாதிக்கப்பட்ட இலைகள் மற்றும் கிளைகளைக் குறித்தே வெட்டுங்கள், மற்றும் செடிக்குப் பின்புறத்தில் நீர் கொடுக்கவும், மேல் நீர்மாரி செய்து விடாதீர்கள்."
    ),
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': (
        "கிரே லீஃப் ஸ்பாட்டை ஸ்ட்ரோபிலுரின்கள் (எ.கா. அசோக்ஸிஸ்ட்ரோபின்) அல்லது டிரையஸோல்கள் (எ.கா. ப்ரோபிகோனசோல்) போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி நிர்வகிக்கலாம். "
        "பயிர் கழிவுகளை அகற்றி, நோய் பரவலைக் குறைக்கும் வகையில் பயிர் சுழற்சியைப் பயன்படுத்துங்கள். எதிர்ப்புடைய ஹைபிரிட்கள் பயிரிடுவதும் பரிந்துரைக்கப்படுகிறது."
    ),
    'Corn_(maize)Common_rust': (
        "காமன் ரஸ்டை மான்கோஸெப் அல்லது குளோரோத்தலோநில் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி கட்டுப்படுத்தலாம். "
        "சரியான பயிர் சுழற்சியை உறுதி செய்து, காற்றோட்டத்தை மேம்படுத்த உயர்ந்த செடிகள் அடர்த்தியைத் தவிர்க்கவும். "
        "ரஸ்ட்டுக்கு எதிர்ப்புடைய ஹைபிரிட்களைப் பயன்படுத்துவது மிகச் சிறந்த தடுப்பு நடவடிக்கைகளில் ஒன்றாகும்."
    ),
    'Corn_(maize)_Northern_Leaf_Blight': (
        "நார்தர்ன் லீஃப் ப்ளைட்டை ஸ்ட்ரோபிலுரின்கள் (அசோக்ஸிஸ்ட்ரோபின்) மற்றும் டிரையஸோல்கள் (தேபுகோனசோல்) போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி கட்டுப்படுத்தலாம். "
        "நோய் சுழற்சியை முறியடிக்க எதிர்ப்புடைய ஹைபிரிட்கள் பயிரிட்டு, பயிர் சுழற்சியைப் பயன்படுத்துங்கள். "
        "நோயின் தீவிரத்தை அதிகரிக்க, வயல்களுக்கு அதிகமான நீர்ப்பாசனம் அளிக்க வேண்டாம்."
    ),
    'Grape___Black_rot': (
        "திராட்சையில் கருப்பு சிதைவைக் கட்டுப்படுத்த, காப்டான், மைக்லோபுடேநில் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளை தொடர்ந்து பயன்படுத்துவது அவசியம். "
        "பாதிக்கப்பட்ட இலைகள் மற்றும் பழங்களை உடனடியாக அகற்றி, காற்றோட்டத்தை அதிகரிக்க சரியான வெட்டுமுறையை உறுதி செய்யுங்கள். "
        "வளர்ச்சிகாலத்தின் ஆரம்பத்தில் பூஞ்சிமருந்து சிகிச்சையைத் தொடங்கவும், குறிப்பாக மழைக்காலங்களில் தொடரவும்."
    ),
    'Grape__Esca(Black_Measles)': (
        "கறுப்பு மீஸில்ஸ் நோய்க்குச் சிகிச்சை இல்லை. சரியான நீர்ப்பாசனத்தை உறுதி செய்து, கற்றையில் காயங்களைத் தவிர்க்க வேண்டும், இது பூஞ்சை நுழைய உதவும். "
        "நீங்கள் பாதுகாப்பு சிகிச்சையாக தியோஃபனேட்-மிதைல் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்த முடியும், ஆனால் இது ஏற்கனவே பாதிக்கப்பட்ட செடிகளுக்குச் சிகிச்சையளிக்காது. "
        "நோய்த்தொற்று மிகுந்த திராட்சை செடிகளை அகற்றி அழித்து, நோய் பரவலைத் தடையுங்கள்."
    ),
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': (
        "இலை சிதைவைக் காப்பர் அடிப்படையிலான பூஞ்சிமருந்து தெளிப்புகள் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி சிகிச்சையளிக்கலாம். "
        "வழக்கமான வெட்டுமுறையை உறுதி செய்து காற்றோட்டத்தை மேம்படுத்துங்கள், மேலும் பாதிக்கப்பட்ட இலைகள் அல்லது பழங்களை அகற்றுங்கள். "
        "மேல்நீர்மாரி செய்வதைத் தவிர்க்கவும், இலைகளில் ஈரப்பதத்தை குறைக்க துளி நீர்மாரியை விரும்பிப் பயன்படுத்துங்கள்."
    ),
    'Orange__Haunglongbing(Citrus_greening)': (
        "சிட்ரஸ் கிரீனிங் ஒரு தீவிர நோயாகும், இதற்குச் சிகிச்சை இல்லை. பாதிக்கப்பட்ட மரங்களை அகற்றி அழிக்க வேண்டும். "
        "நோயின் உயிரியாக உள்ள ஆசிய சிட்ரஸ் சில்லிடை (Asian citrus psyllid) இமிடாக்ளோப்ரிட் அல்லது தியாமெதோக்ஸாம் போன்ற பூச்சிக்கொல்லிகளைப் பயன்படுத்தி கட்டுப்படுத்தவும். "
        "சரியான தோட்ட பராமரிப்பு நடைமுறைகளை நடைமுறைப்படுத்துங்கள், இதில் நோய்த்தொற்றுக்குள்ளான செடிகளை அடிக்கடி கண்காணித்து அகற்றுவது அடங்கும்."
    ),
    'Peach___Bacterial_spot': (
        "பாக்டீரியல் ஸ்பாட்டை காப்பர் அடிப்படையிலான பூஞ்சிமருந்துகள் அல்லது ஆக்ஸிடெட்ராசைக்கிளின் தெளிப்புகளைப் பயன்படுத்தி கட்டுப்படுத்தலாம். "
        "பாதிக்கப்பட்ட இலைகள் மற்றும் பழங்களை அகற்றி அழிக்கவும். மேல்நீர்மாரி செய்ய வேண்டாம், ஏனெனில் ஈரப்பதம் பாக்டீரியாவின் பரவலை அதிகரிக்கச் செய்யும். "
        "புதிய பீச் மரங்களை நட்டபோது நோய்க்கு எதிர்ப்புடைய வகைகளைத் தேர்வு செய்து, மரங்களை வெட்டிக் காற்றோட்டத்தை மேம்படுத்துங்கள்."
    ),
    'Pepper,bell__Bacterial_spot': (
        "பாக்டீரியல் ஸ்பாட்டை காப்பர் அடிப்படையிலான பாக்டீரியசைடுகளைச் சரியாகப் பயன்படுத்தி கட்டுப்படுத்தலாம். "
        "மேல்நீர்மாரி செய்வதைத் தவிர்த்து, சரியான பயிர் சுழற்சியை உறுதி செய்யுங்கள். "
        "நோய்த்தொற்றுக்குள்ளான இலைகள் மற்றும் பழங்களை உடனடியாக அகற்றி, நோய் பரவலைத் தடையுங்கள்."
    ),
    'Potato___Early_blight': (
        "ஆரம்ப நிலை பிளைட்டிற்கு, குளோரோத்தலோநில் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்துங்கள். "
        "பயிர் சுழற்சி செய்யுங்கள், மேல்நீர்மாரி செய்வதைத் தவிர்க்கவும், மற்றும் செடிகளுக்குள் சரியான இடைவெளியை உறுதி செய்து ஈரப்பதத்தைக் குறையச் செய்யுங்கள். "
        "நோய்த்தொற்றுக்குள்ளான செடி கழிவுகளை அகற்றி அழிக்கவும், நோய் பரவலைக் குறைக்கும்."
    ),
    'Potato___Late_blight': (
        "பின்பட்ட பிளைட்டிற்கு, காப்பர் அடிப்படையிலான பொருட்கள் அல்லது மெஃபெனோக்சாம் போன்ற முறைமையான பூஞ்சிமருந்துகளைப் பயன்படுத்தி நிர்வகிக்கலாம். "
        "நோய்த்தொற்றுக்குள்ளான செடிகள் மற்றும் கிழங்குகளை அழித்து, நோய் பரவலைத் தடையுங்கள். மேல்நீர்மாரி செய்வதைத் தவிர்க்கவும், மற்றும் சரியான மண்ணின் வடிகட்டுதலை உறுதி செய்யுங்கள்."
    ),
    'Squash___Powdery_mildew': (
        "பவுட்ரி மில்டியூக்கு, சல்பர், நீம் எண்ணெய் அல்லது பொட்டாசியம் பைக்கார்பனேட் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்துங்கள். "
        "மிகவும் பாதிக்கப்பட்ட இலைகளை அகற்றுங்கள், காற்றோட்டத்தை மேம்படுத்துங்கள், மற்றும் செடிகளின் அடிப்பகுதியில் மட்டுமே நீர் கொடுக்கவும், இலைகளை ஈரமாகச் செய்யாமல் இருக்கவும். "
        "செடிகளை மிகுந்த அடர்த்தியாக வளர்ப்பதைத் தவிர்க்கவும், ஏனெனில் இது மில்டியூ வளர்ச்சியை ஊக்குவிக்கிறது."
    ),
    'Strawberry___Leaf_scorch': (
        "இலை சூரியகாயத்தை காப்டான் அல்லது காப்பர் அடிப்படையிலான தெளிப்புகள் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி சிகிச்சையளிக்கலாம். "
        "பாதிக்கப்பட்ட இலைகளை அகற்றவும், மற்றும் செடிகளுக்குள் சரியான இடைவெளியை உறுதி செய்து காற்றோட்டத்தை மேம்படுத்துங்கள். மேல்நீர்மாரி செய்வதைத் தவிர்க்கவும்."
    ),
    'Tomato___Bacterial_spot': (
        "பாக்டீரியல் ஸ்பாட்டை காப்பர் அடிப்படையிலான பாக்டீரியசைடுகள் அல்லது ஸ்ட்ரெப்டோமைசின் மூலம் கட்டுப்படுத்தலாம். "
        "பாதிக்கப்பட்ட இலைகள் மற்றும் பழங்களை அகற்றி, பாக்டீரியாவின் பரவலைத் தடுக்கும் வகையில் ஈரமான செடிகளுடன் வேலை செய்வதைத் தவிர்க்கவும்."
    ),
    'Tomato___Early_blight': (
        "ஆரம்ப நிலை பிளைட்டை குளோரோத்தலோநில், காப்பர் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளைக் கொண்டு நிர்வகிக்கலாம். "
        "கீழ்தர இலைகளை வெட்டிக் காற்றோட்டத்தை மேம்படுத்தவும், மண்ணிலிருந்து இலைகளில் சிதறாதிருக்க முடிச்சுகளைப் பயன்படுத்தவும், மற்றும் பாதிக்கப்பட்ட இலைகளை அகற்றவும்."
    ),
    'Tomato___Late_blight': (
        "பின்பட்ட பிளைட்டை காப்பர் அடிப்படையிலான பொருட்கள் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி சிகிச்சையளிக்கலாம். "
        "நோய்த்தொற்றுக்குள்ளான செடிகளை அகற்றி அழித்து, மேல்நீர்ப்பாசனத்தைத் தவிர்க்கவும். சாத்தியமான அளவில் நோய்க்கு எதிர்ப்புடைய வகைகளைப் பயன்படுத்தவும்."
    ),
    'Tomato___Leaf_Mold': (
        "இலை பூஞ்சையைக் குளோரோத்தலோநில் அல்லது மான்கோஸெப் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி கட்டுப்படுத்தலாம். "
        "சிறந்த காற்றோட்டத்தை உறுதி செய்து, மேல்நீர்மாரி செய்வதைத் தவிர்க்கவும், மேலும் மிகுந்த பாதிக்கப்பட்ட இலைகளை வெட்டுங்கள்."
    ),
    'Tomato___Septoria_leaf_spot': (
        "செப்டோரியா இலை தழும்பை மான்கோஸெப் அல்லது குளோரோத்தலோநில் போன்ற பூஞ்சிமருந்துகளைக் கொண்டு சிகிச்சையளிக்கலாம். "
        "பாதிக்கப்பட்ட பகுதிகளை வெட்டி அகற்றுங்கள் மற்றும் மண்ணிலிருந்து நோய் பரவலைக் குறைக்க முடிச்சுகளைப் பயன்படுத்துங்கள்."
    ),
    'Tomato___Spider_mites Two-spotted_spider_mite': (
        "ஸ்பைடர் மைட்ஸைக் கட்டுப்படுத்த ஈரப்பதத்தை அதிகரித்து, பூச்சிக்கொல்லி சோப்பு அல்லது நீம் எண்ணெயைத் தெளிக்கலாம். "
        "பூச்சிகளைக் களைவதற்காகச் செடியில் முறையாக நீர்த்தேக்கத்தைச் செய்து, பருவமுதிர்ந்த பூச்சிகளைக் கட்டுப்படுத்தும் பூச்சிகளைப் பயன்படுத்தலாம்."
    ),
    'Tomato___Target_Spot': (
        "டார்கெட் ஸ்பாட்டை குளோரோத்தலோநில் அல்லது காப்பர் அடிப்படையிலான பொருட்கள் போன்ற பூஞ்சிமருந்துகளைப் பயன்படுத்தி சிகிச்சையளிக்கலாம். "
        "சரியான இடைவெளியை வைத்துக் காற்றோட்டத்தை மேம்படுத்தவும், பாதிக்கப்பட்ட பகுதிகளை வெட்டி அகற்றவும், மேல்நீர்ப்பாசனத்தைத் தவிர்க்கவும்."
    ),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
        "இந்த வைரஸிற்கு சிகிச்சை இல்லை. நோய்த்தொற்றுக்குள்ளான செடிகளை உடனடியாக அகற்ற வேண்டும், நோய் பரவலைத் தடுக்கும் வகையில். "
        "வைட்ட்ப்ளைஸ் என்ற பூச்சியை இமிடாக்ளோப்ரிட் அல்லது நீம் எண்ணெய் போன்ற பூச்சிக்கொல்லிகளைப் பயன்படுத்தி கட்டுப்படுத்துங்கள். "
        "சாத்தியமான அளவில் நோய்க்கு எதிர்ப்புடைய வகைகளைப் பயன்படுத்துங்கள்."
    ),
    'Tomato___Tomato_mosaic_virus': (
        "இந்த வைரஸிற்கு சிகிச்சை இல்லை. நோய்த்தொற்றுக்குள்ளான செடிகளை அகற்றி, நோய்த்தொற்றுக்குள்ளான செடிகளுடன் வேலை செய்தபிறகு கருவிகளைச் சுத்தம் செய்யுங்கள். "
        "செடிகள் ஈரமாக இருக்கும்போது அவற்றைக் கையாள்வதைத் தவிர்க்கவும், ஏனெனில் வைரஸ் எளிதாகத் தொற்றுகிறது."
    )}
        },
        'Hindi': {
            'title': "पादप रोग वर्गीकरणकर्ता",
            'upload_prompt': "पौधे की पत्ती की तस्वीर अपलोड करें।",
            'choose_image': "छवि चुनें...",
            'predicted_class': "अनुमानित श्रेणी",
            'confidence': "विश्वास",
            'solution' : {
    'Apple___Apple_scab': (
        "Apple scab एक फंगल रोग है। कैप्टान या मायक्लोबुटानिल युक्त फफूंदनाशकों का प्रयोग करें।"
        "फफूंद के प्रसार को कम करने के लिए गिरे हुए पत्तों या फलों को हटा दें और नष्ट कर दें।"
        "वायु संचार को बेहतर बनाने के लिए पेड़ों की उचित छंटाई सुनिश्चित करें। बढ़ते मौसम के दौरान नियमित रूप से छिड़काव करना महत्वपूर्ण है।"
    ),

    'Apple___Black_rot': (
        "संक्रमित शाखाओं और फलों को हटाकर Black rot को नियंत्रित किया जा सकता है। कॉपर-आधारित फफूंदनाशकों या लिक्विड लाइम सल्फर का प्रयोग करें।"
        "वायु संचार को बेहतर बनाने और भीड़भाड़ से बचने के लिए पेड़ों की छंटाई करें। बढ़ते मौसम के दौरान, कैप्टान या थियोफैनेट-मिथाइल जैसे फफूंदनाशकों का छिड़काव करने की सलाह दी जाती है।"
    ),

    'Apple___Cedar_apple_rust': (
        "Cedar apple rust का सबसे अच्छा प्रबंधन पंखुड़ी गिरने पर मायक्लोबुटानिल या प्रोपिकोनाज़ोल युक्त फफूंदनाशकों का प्रयोग करके और हर 7-10 दिनों में दोहराकर किया जा सकता है।"
        "आस-पास के किसी भी देवदार के पेड़ को हटा दें, क्योंकि वे रोग के वैकल्पिक मेजबान हैं। नए पेड़ लगाते समय प्रतिरोधी सेब किस्मों का उपयोग करें।"
    ),

    'Cherry_(including_sour)_Powdery_mildew': (
        "Powdery mildew के लिए, सल्फर या पोटेशियम बाइकार्बोनेट कवकनाशी प्रभावी हैं।"
        "पौधे के चारों ओर उचित वायु परिसंचरण सुनिश्चित करें और अत्यधिक नाइट्रोजन निषेचन से बचें, जो नरम, संवेदनशील वृद्धि को बढ़ावा दे सकता है।"
        "प्रभावित पत्तियों और शाखाओं की छंटाई करें, तथा ऊपर से पानी देने के बजाय पौधे के आधार पर पानी दें।"
    ),

    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': (
        "Gray leaf spot को स्ट्रोबिलुरिन (जैसे, Azoxystrobin) या ट्रायज़ोल (जैसे, Propiconazole) जैसे कवकनाशकों का उपयोग करके प्रबंधित किया जा सकता है।"
        "फसल अवशेषों को हटाएँ और रोगजनक कैरीओवर को कम करने के लिए फसलों को घुमाएँ। प्रतिरोधी संकर लगाने की भी सिफारिश की जाती है।"
    ),

    'Corn_(maize)_Common_rust': (
        "Common rust को मैन्कोज़ेब या क्लोरोथालोनिल जैसे कवकनाशकों से नियंत्रित किया जा सकता है।"
        "वायु परिसंचरण में सुधार के लिए उचित फसल चक्र सुनिश्चित करें और उच्च पौधे घनत्व से बचें।"
        "जंग प्रतिरोधी संकर का उपयोग करना सबसे प्रभावी निवारक उपायों में से एक है।"
    ),

    'Corn_(maize)_Northern_Leaf_Blight': (
        "Northern leaf blight को स्ट्रोबिलुरिन (Azoxystrobin) और ट्रायज़ोल (Tebuconazole) जैसे कवकनाशकों से नियंत्रित किया जा सकता है।"
        "रोग चक्र को तोड़ने के लिए प्रतिरोधी संकर पौधे लगाएँ और फसल चक्र का उपयोग करें।"
        "सुनिश्चित करें कि खेतों में अत्यधिक सिंचाई न की जाए, क्योंकि नमी से रोग की गंभीरता बढ़ सकती है।"
    ),

    'Grape___Black_rot': (
        "Grape में Black rot के लिए कैप्टान, मायक्लोबुटानिल या मैन्कोज़ेब सहित लगातार कवकनाशी के इस्तेमाल की आवश्यकता होती है।"
        "संक्रमित पत्तियों और फलों को तुरंत हटा दें, और हवा के संचार को बढ़ाने के लिए उचित छंटाई सुनिश्चित करें।"
        "बढ़ते मौसम में कवकनाशी उपचार शुरू करें और पूरे मौसम में जारी रखें, खासकर गीली अवधि के दौरान।"
    ),

    'Grape__Esca(Black_Measles)': (
        "दुर्भाग्य से, Black Measles का कोई इलाज नहीं है। उचित सिंचाई सुनिश्चित करके और बेल को चोट पहुँचाने से बचाकर रोकथाम पर ध्यान दें, जिससे कवक अंदर जा सकता है।"
        "आप सुरक्षात्मक उपचार के रूप में थियोफैनेट-मिथाइल जैसे कवकनाशी का उपयोग करने पर विचार कर सकते हैं, हालाँकि यह पहले से संक्रमित पौधों को ठीक नहीं करेगा।"
        "बीमारी को फैलने से रोकने के लिए गंभीर रूप से संक्रमित बेलों को हटा दें और नष्ट कर दें।"
    ),

    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': (
        "Leaf blight का उपचार कॉपर-आधारित स्प्रे या मैन्कोज़ेब जैसे कवकनाशकों से किया जा सकता है।"
        "हवा के संचार की अनुमति देने के लिए नियमित छंटाई सुनिश्चित करें, और किसी भी संक्रमित पत्तियों या फलों को हटा दें।"
        "ऊपर से सिंचाई करने से बचें और पत्तियों पर नमी कम करने के लिए ड्रिप सिंचाई का विकल्प चुनें।"
    ),

    'Orange__Haunglongbing(Citrus_greening)': (
        "Citrus greening एक गंभीर बीमारी है जिसका कोई ज्ञात इलाज नहीं है। संक्रमण को फैलने से रोकने के लिए संक्रमित पेड़ों को हटाकर नष्ट कर देना चाहिए।"
        "रोग के कीट वाहक एशियाई साइट्रस साइलीड को इमिडाक्लोप्रिड या थियामेथोक्सम जैसे कीटनाशकों से नियंत्रित करें।"
        "संक्रमित पौधों की नियमित निगरानी और हटाने सहित उचित बाग प्रबंधन प्रथाओं को लागू करें।"
    ),

    'Peach___Bacterial_spot': (
        "Bacterial spot को कॉपर-आधारित कवकनाशी या ऑक्सीटेट्रासाइक्लिन स्प्रे का उपयोग करके नियंत्रित किया जा सकता है।"
        "संक्रमित पत्तियों और फलों को हटाकर नष्ट कर दें। ऊपर से पानी देने से बचें, क्योंकि नमी बैक्टीरिया के प्रसार को बढ़ावा देती है।"
        "नए आड़ू के पेड़ लगाते समय प्रतिरोधी किस्मों का चयन करें और हवा के संचार को बेहतर बनाने के लिए पेड़ों की छंटाई करें।"
    ),

    'Pepper,bell__Bacterial_spot': (
        "Bacterial spot को कॉपर-आधारित जीवाणुनाशकों को नियमित रूप से लगाकर नियंत्रित करें।"
        "ऊपर से पानी देने से बचें और उचित फसल चक्र सुनिश्चित करें। आगे प्रसार को रोकने के लिए संक्रमित पत्तियों और फलों को तुरंत हटा दें।"
    ),

    'Potato___Early_blight': (
        "Early blight के लिए, क्लोरोथालोनिल या मैन्कोज़ेब जैसे कवकनाशी का प्रयोग करें।"
        "फसलों को घुमाएँ, ऊपर से पानी देने से बचें, और नमी को कम करने के लिए पौधों के बीच उचित दूरी सुनिश्चित करें।"
        "रोग के प्रसार को सीमित करने के लिए संक्रमित पौधों के मलबे को हटाएँ और नष्ट करें।"
    ),

    'Potato___Late_blight': (
        "Late blight को कॉपर-आधारित उत्पादों या मेफेनोक्सम जैसे प्रणालीगत कवकनाशी का उपयोग करके प्रबंधित किया जा सकता है।"
        "प्रसार को रोकने के लिए संक्रमित पौधों और कंदों को नष्ट करें। ऊपर से पानी देने से बचें और उचित मिट्टी की जल निकासी सुनिश्चित करें।"
    ),

    'Squash___Powdery_mildew': (
        "Powdery mildew के लिए, सल्फर, नीम तेल या पोटेशियम बाइकार्बोनेट जैसे कवकनाशी का प्रयोग करें।"
        "गंभीर रूप से संक्रमित पत्तियों को हटाएँ, वायु परिसंचरण में सुधार करें, और पत्तियों को गीला होने से बचाने के लिए आधार पर पौधों को पानी दें।"
        "पौधों को अधिक भीड़भाड़ से बचाएँ, क्योंकि इससे फफूंदी के विकास को बढ़ावा मिलता है।"
    ),

    'Strawberry___Leaf_scorch': (
        "Leaf scorch का उपचार कैप्टान या कॉपर-आधारित स्प्रे जैसे कवकनाशी से किया जा सकता है।"
        "संक्रमित पत्तियों को हटा दें और हवा के संचार को बेहतर बनाने के लिए पौधों के बीच उचित दूरी सुनिश्चित करें। ऊपर से पानी देने से बचें।"
    ),

    'Tomato___Bacterial_spot': (
        "Bacterial spot को कॉपर-आधारित जीवाणुनाशकों या स्ट्रेप्टोमाइसिन का उपयोग करके नियंत्रित किया जा सकता है।"
        "ऊपर से पानी देने से बचें और संक्रमित पत्तियों और फलों को तुरंत हटा दें।"
        "रोग प्रतिरोधी किस्मों का उपयोग करें और हवा के संचार को बेहतर बनाने के लिए उचित छंटाई सुनिश्चित करें।"
    ),

    'Tomato___Early_blight': (
        "Early blight को नियंत्रित करने के लिए क्लोरोथालोनिल या मैनकोज़ेब जैसे कवकनाशी का छिड़काव करें।"
        "प्रभावित पत्तियों को हटा दें, पौधों के बीच पर्याप्त दूरी बनाए रखें और नमी के निर्माण को रोकने के लिए पत्तियों के बजाय मिट्टी को पानी दें।"
    ),

    'Tomato___Late_blight': (
        "Late blight को मेटालैक्सिल या क्लोरोथालोनिल जैसे कवकनाशकों का प्रयोग करके नियंत्रित किया जा सकता है।"
        "प्रभावित पौधों और कंदों को हटा दें और नष्ट कर दें। ऊपर से पानी देने से बचें और सिंचाई के बीच मिट्टी के सूखने की अनुमति दें।"
    ),

    'Tomato___Leaf_Mold': (
        "Leaf mold के लिए, क्लोरोथालोनिल या ताम्र कवकनाशी का उपयोग करें।"
        "ऊपर से पानी देने से बचें और पत्तियों पर नमी को कम करने के लिए पौधों के चारों ओर उचित वायु परिसंचरण सुनिश्चित करें।"
    ),

    'Tomato___Septoria_leaf_spot': (
        "Septoria leaf spot को नियंत्रित करने के लिए क्लोरोथालोनिल या ताम्र कवकनाशी का उपयोग करें।"
        "संक्रमित पत्तियों को हटा दें और सुनिश्चित करें कि पौधों के बीच पर्याप्त दूरी हो ताकि उचित वायु परिसंचरण हो सके।"
        "सिंचाई करते समय पौधे के आधार पर पानी दें और ऊपर से पानी देने से बचें।"
    ),

    'Tomato___Spider_mites Two-spotted_spider_mite': (
        "Spider mites को नियंत्रित करने के लिए साबुन के स्प्रे या नीम तेल का प्रयोग करें।"
        "पौधों की नियमित निगरानी करें और संक्रमण के शुरुआती संकेत दिखाने वाले किसी भी पत्ते को हटा दें।"
        "पौधों को तनाव से बचाएँ, क्योंकि कमजोर पौधे अधिक संवेदनशील होते हैं।"
    ),

    'Tomato___Target_Spot': (
        "Target spot के लिए, कवकनाशी का नियमित छिड़काव करें जैसे क्लोरोथालोनिल, कैप्टान या मैनकोज़ेब।"
        "संक्रमित पत्तियों को हटा दें, और वायु परिसंचरण में सुधार करने के लिए पौधों को ठीक से छाँटें।"
        "ऊपर से पानी देने से बचें, क्योंकि नमी संक्रमण को बढ़ावा दे सकती है।"
    ),

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
        "Tomato Yellow Leaf Curl Virus के वाहक सफेद मक्खियों को नियंत्रित करने के लिए इमिडाक्लोप्रिड या बिफेंथ्रिन जैसे कीटनाशकों का उपयोग करें।"
        "रोग को फैलने से रोकने के लिए संक्रमित पौधों को हटा दें और नष्ट कर दें।"
        "प्रतिरोधी टमाटर की किस्मों का उपयोग करें और बाग में अच्छे कीट प्रबंधन प्रथाओं को लागू करें।"
    ),

    'Tomato___Tomato_mosaic_virus': (
        "Tomato mosaic virus का कोई इलाज नहीं है। संक्रमित पौधों को तुरंत हटा दें और नष्ट कर दें।"
        "रोग फैलाने से रोकने के लिए उपकरणों और हाथों को नियमित रूप से साफ करें।"
        "प्रतिरोधी किस्मों का उपयोग करें और पौधों की निगरानी करें ताकि प्रारंभिक लक्षणों का पता लगाया जा सके।"
    )}
        },
        'Telugu': {
            'title': "మొక్కల వ్యాధి వర్గీకరణ",
            'upload_prompt': "మీ మొక్క ఆకుని అప్‌లోడ్ చేయండి.",
            'choose_image': "చిత్రాన్ని ఎంచుకోండి...",
            'predicted_class': "అంచనా వర్గం",
            'confidence': "ఆత్మవిశ్వాసం",
            'solution' : {
    'Apple___Apple_scab': (
        "యాపిల్ స్కాబ్ అనేది శిలీంధ్ర వ్యాధి. క్యాప్టాన్ లేదా మైక్లోబుటానిల్ కలిగిన శిలీంద్రనాశకాలను వర్తించండి."
        "ఫంగస్ వ్యాప్తిని తగ్గించడానికి ఏవైనా పడిపోయిన ఆకులు లేదా పండ్లను తొలగించి నాశనం చేయండి."
        "వాయు ప్రసరణను మెరుగుపరచడానికి సరైన చెట్ల కత్తిరింపును నిర్ధారించుకోండి. పెరుగుతున్న కాలంలో రెగ్యులర్ అప్లికేషన్లు చాలా ముఖ్యమైనవి."
    ),
    'Apple___Black_rot': (
        "నల్ల తెగులు సోకిన కొమ్మలు మరియు పండ్లను తొలగించడం ద్వారా నియంత్రించవచ్చు. రాగి ఆధారిత శిలీంధ్రనాశకాలు లేదా ద్రవ సున్నం సల్ఫర్‌ను వర్తించండి."
        "వాయు ప్రసరణను మెరుగుపరచడానికి మరియు రద్దీని నివారించడానికి చెట్లను కత్తిరించండి. పెరుగుతున్న కాలంలో, క్యాప్టాన్ లేదా థియోఫనేట్-మిథైల్ వంటి శిలీంద్రనాశకాలు సిఫార్సు చేయబడతాయి."
    ),
    'Apple___Cedar_apple_rust': (
        "సెడార్ యాపిల్ రస్ట్‌ను మైక్లోబుటానిల్ లేదా ప్రొపికోనాజోల్ కలిగిన శిలీంద్రనాశకాలను పూయడం మరియు ప్రతి 7-10 రోజులకు పునరావృతం చేయడం ద్వారా ఉత్తమంగా నిర్వహించవచ్చు."
        "దేవదారు చెట్లు సమీపంలో ఉంటే తొలగించండి, అవి వ్యాధికి ప్రత్యామ్నాయ హోస్ట్‌లు. కొత్త చెట్లను నాటేటప్పుడు నిరోధక ఆపిల్ రకాలను ఉపయోగించండి."
    ),
    'Cherry_(including_sour)_Powdery_mildew': (
        "పొడి బూజు కోసం సల్ఫర్ లేదా పొటాషియం బైకార్బోనేట్ వంటి శిలీంధ్రనాశకాలు ప్రభావవంతంగా ఉంటాయి."
        "మొక్క చుట్టూ సరైన గాలి ప్రసరణను నిర్ధారించండి మరియు అధిక నత్రజని ఫలదీకరణం నివారించండి, ఇది మృదువైన, గ్రహణశీల పెరుగుదలను ప్రోత్సహిస్తుంది."
        "బాధిత ఆకులు మరియు కొమ్మలను కత్తిరించండి, మరియు మొక్కల అడుగుభాగంలో నీటిని కాకుండా కిందినుండి నీరు పెట్టండి."
    ),
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': (
        "బూడిద ఆకు మచ్చను అజోక్సిస్ట్రోబిన్ వంటి స్ట్రోబిలురిన్స్ లేదా ప్రొపికోనాజోల్ వంటి ట్రైజోల్స్‌తో నిర్వహించవచ్చు."
        "పాథోజెన్ క్యారీఓవర్‌ను తగ్గించడానికి పంట అవశేషాలను తొలగించి పంట భ్రమణాన్ని నిర్వహించండి. నిరోధక హైబ్రిడ్‌లను నాటడం సిఫార్సు చేయబడింది."
    ),
    'Corn_(maize)_Common_rust': (
        "మాంకోజెబ్ లేదా క్లోరోథలోనిల్ వంటి శిలీంద్రనాశకాలతో సాధారణ తుప్పును నియంత్రించవచ్చు."
        "సరైన పంట భ్రమణాన్ని నిర్ధారించుకోండి మరియు గాలి ప్రసరణను మెరుగుపరచడానికి అధిక మొక్కల సాంద్రతను నివారించండి."
        "రస్ట్-రెసిస్టెంట్ హైబ్రిడ్లను ఉపయోగించడం అత్యంత ప్రభావవంతమైన నివారణ చర్యలలో ఒకటి."
    ),
    'Corn_(maize)_Northern_Leaf_Blight': (
        "ఉత్తర ఆకు ముడతను అజోక్సిస్ట్రోబిన్ వంటి స్ట్రోబిలురిన్స్ మరియు టెబుకోనాజోల్ వంటి ట్రైజోల్స్‌తో నియంత్రించవచ్చు."
        "నిరోధక హైబ్రిడ్లను నాటండి మరియు వ్యాధి చక్రాన్ని విచ్ఛిన్నం చేయడానికి పంట భ్రమణాన్ని ఉపయోగించండి."
        "తేమ వ్యాధి యొక్క తీవ్రతను పెంచుతుంది కాబట్టి పొలాలు ఎక్కువ నీరు త్రాగకుండా చూసుకోండి."
    ),
    'Grape___Black_rot': (
        "ద్రాక్షలో నలుపు తెగులుకు క్యాప్టాన్, మైక్లోబుటానిల్ లేదా మాంకోజెబ్ వంటి స్థిరమైన శిలీంద్రనాశకాలను వర్తించాలి."
        "సోకిన ఆకులు మరియు పండ్లను వెంటనే తొలగించి, గాలి ప్రసరణను పెంచడానికి సరైన కత్తిరింపును నిర్ధారించండి."
        "ఎదుగుదల సీజన్ ప్రారంభంలో శిలీంద్రనాశకాలను పూయడం ప్రారంభించి, ముఖ్యంగా తడి కాలంలో, సీజన్ అంతటా కొనసాగించండి."
    ),
    'Grape__Esca(Black_Measles)': (
        "దురదృష్టవశాత్తూ, బ్లాక్ మీజిల్స్‌కు ఎటువంటి నివారణ లేదు. సరైన నీటిపారుదలని నిర్ధారించడం ద్వారా మరియు తీగల గాయాలను నివారించడం ద్వారా నివారణపై దృష్టి పెట్టండి."
        "థియోఫానేట్-మిథైల్ వంటి శిలీంద్రనాశకాలను రక్షక చికిత్సగా ఉపయోగించవచ్చు, అయితే ఇవి ఇప్పటికే సోకిన మొక్కలను నయం చేయవు."
        "తీవ్రంగా సోకిన తీగలను తొలగించి, వ్యాధి వ్యాప్తి చెందకుండా నివారించండి."
    ),
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': (
        "ఆకు ముడతను రాగి ఆధారిత స్ప్రేలు లేదా మాంకోజెబ్ వంటి శిలీంద్రనాశకాలతో చికిత్స చేయవచ్చు."
        "వాయు ప్రసరణను మెరుగుపరచడానికి క్రమం తప్పకుండా కత్తిరించడం నిర్ధారించుకోండి, మరియు సోకిన ఆకులు లేదా పండ్లను తొలగించండి."
        "ఆకులపై తేమను తగ్గించడానికి ఓవర్ హెడ్ నీటిపారుదలని నివారించండి, మరియు బిందు సేద్యాన్ని ఎంచుకోండి."
    ),
    'Orange__Haunglongbing(Citrus_greening)': (
        "సిట్రస్ గ్రీన్నింగ్ అనేది ఎటువంటి నివారణ లేని తీవ్రమైన వ్యాధి. సోకిన చెట్లను తొలగించి నాశనం చేయాలి."
        "ఇమిడాక్లోప్రిడ్ లేదా థియామెథోక్సామ్ వంటి క్రిమిసంహారకాలను ఉపయోగించి ఆసియా సిట్రస్ సైలిడ్‌ను నియంత్రించండి."
        "క్రమం తప్పకుండా పర్యవేక్షణ చేసి, సోకిన మొక్కలను తొలగించడంతో సహా సరైన పండ్ల తోట నిర్వహణ పద్ధతులను అమలు చేయండి."
    ),
    'Peach___Bacterial_spot': (
        "బాక్టీరియల్ స్పాట్‌ను రాగి-ఆధారిత శిలీంద్రనాశకాలు లేదా ఆక్సిటెట్రాసైక్లిన్ స్ప్రేలను ఉపయోగించి నియంత్రించవచ్చు."
        "సోకిన ఆకులు మరియు పండ్లను తొలగించి నాశనం చేయండి. తేమ బ్యాక్టీరియా వ్యాప్తిని ప్రోత్సహిస్తుంది కాబట్టి, ఓవర్ హెడ్ నీరు త్రాగుటను నివారించండి."
        "కొత్త పీచు చెట్లను నాటేటప్పుడు నిరోధక రకాలను ఎంచుకోండి, మరియు గాలి ప్రసరణను మెరుగుపరచడానికి చెట్లను కత్తిరించండి."
    ),
    'Pepper,bell___Bacterial_spot': (
        "బాక్టీరియల్ స్పాట్‌ను రాగి ఆధారిత శిలీంద్రనాశకాలు క్రమం తప్పకుండా వర్తింపజేయడం ద్వారా నియంత్రించవచ్చు."
        "ఓవర్ హెడ్ నీరు త్రాగుట నివారించండి మరియు సరైన పంట భ్రమణాన్ని నిర్ధారించండి. సోకిన ఆకులు మరియు పండ్లను వెంటనే తొలగించి, మరింత వ్యాప్తి చెందకుండా నివారించండి."
    ),
    'Potato___Early_blight': (
        "ప్రారంభ ముడత కోసం క్లోరోథలోనిల్ లేదా మాంకోజెబ్ వంటి శిలీంద్రనాశకాలను వర్తించండి."
        "పంట భ్రమణాన్ని అమలు చేయండి, ఓవర్ హెడ్ నీటిపారుదలని నివారించండి, మరియు తేమను తగ్గించడానికి మొక్కల మధ్య సరైన అంతరం ఉండేలా చూసుకోండి."
        "వ్యాధి వ్యాప్తిని పరిమితం చేయడానికి సోకిన మొక్కల శిధిలాలను తొలగించి నాశనం చేయండి."
    ),
    'Potato___Late_blight': (
        "రాగి ఆధారిత ఉత్పత్తులు లేదా మెఫెనాక్సామ్ వంటి దైహిక శిలీంద్రనాశకాలు ఉపయోగించి ఆలస్య ముడతను నిర్వహించండి."
        "వాతావరణ పరిస్థితులకు అనుగుణంగా శిలీంధ్రనాశకాలను వర్తించడానికి ఖచ్చితమైన పర్యవేక్షణ మరియు గడియార పద్ధతిని అమలు చేయండి."
        "తేమ మరియు అధిక సాపేక్ష ఆర్ద్రత ఆలస్య ముడతను ప్రోత్సహిస్తుంది, కాబట్టి సరైన గాలి ప్రవాహం మరియు నీటి నిర్వహణను నిర్ధారించుకోండి."
    ),
     'Squash___Powdery_mildew': (
     "బూజు తెగులు కోసం, సల్ఫర్, వేప నూనె లేదా పొటాషియం బైకార్బోనేట్ వంటి శిలీంద్రనాశకాలను వర్తించండి."
     "తీవ్రంగా సోకిన ఆకులను తొలగించండి, గాలి ప్రసరణను మెరుగుపరచండి మరియు ఆకులను తడి చేయకుండా ఉండటానికి బేస్ వద్ద నీటి మొక్కలు వేయండి."
     "మొక్కలు అధికంగా ఉండటం మానుకోండి, ఇది బూజు పెరుగుదలను ప్రోత్సహిస్తుంది."
     ),
      'Strawberry___Leaf_scorch': (
     "కాప్టాన్ లేదా రాగి-ఆధారిత స్ప్రేలు వంటి శిలీంద్రనాశకాలతో ఆకు మంటను చికిత్స చేయవచ్చు."
     "సోకిన ఆకులను తొలగించండి మరియు గాలి ప్రసరణను మెరుగుపరచడానికి మొక్కల మధ్య సరైన అంతరం ఉండేలా చూసుకోండి. ఓవర్ హెడ్ నీరు త్రాగుట నివారించండి."
     ),
    'Tomato___Bacterial_spot':  (
     "బాక్టీరియల్ స్పాట్‌ను రాగి-ఆధారిత బాక్టీరిసైడ్‌లు లేదా స్ట్రెప్టోమైసిన్ ఉపయోగించి నియంత్రించవచ్చు."
     "సోకిన ఆకులు మరియు పండ్లను తొలగించండి మరియు బ్యాక్టీరియా వ్యాప్తిని నిరోధించడానికి తడి మొక్కలతో పని చేయకుండా ఉండండి."
     ),
     'Tomato___Early_blight': (
     "ప్రారంభ ముడతను క్లోరోథలోనిల్, కాపర్ లేదా మాంకోజెబ్ వంటి శిలీంద్రనాశకాలతో నిర్వహించవచ్చు."
     "వాయు ప్రసరణను మెరుగుపరచడానికి దిగువ ఆకులను కత్తిరించండి, ఆకులపై నేల చిమ్మకుండా నిరోధించడానికి రక్షక కవచం మరియు సోకిన ఆకులను తొలగించండి."
     ),
      'Tomato___Late_blight': (
     "లేట్ బ్లైట్ రాగి-ఆధారిత ఉత్పత్తులు లేదా మాంకోజెబ్ వంటి శిలీంద్రనాశకాలను ఉపయోగించి చికిత్స చేయవచ్చు."
     "సోకిన మొక్కలను తొలగించి నాశనం చేయండి మరియు ఓవర్ హెడ్ నీటిపారుదలని నివారించండి. సాధ్యమైన చోట వ్యాధి-నిరోధక రకాలను ఉపయోగించండి."
     ),
       'Tomato___Leaf_Mold': (
     "క్లోరోథలోనిల్ లేదా మాంకోజెబ్ వంటి శిలీంద్రనాశకాలతో ఆకు అచ్చు నియంత్రించబడుతుంది."
     "మంచి గాలి ప్రసరణను నిర్ధారించుకోండి, ఓవర్ హెడ్ నీరు త్రాగుట నివారించండి మరియు భారీగా సోకిన ఆకులను కత్తిరించండి."
     ),
      'Tomato___Septoria_leaf_spot': (
     "సెప్టోరియా లీఫ్ స్పాట్‌ను మాంకోజెబ్ లేదా క్లోరోథలోనిల్ వంటి శిలీంద్రనాశకాలతో చికిత్స చేయవచ్చు."
     "మట్టి నుండి బీజాంశం వ్యాప్తిని తగ్గించడానికి ప్రభావిత ప్రాంతాలను మరియు మల్చ్ మొక్కలను కత్తిరించండి."
     ),
     'Tomato___Spider_mites Two-spotted_spider_mite': (
     "స్పైడర్ మైట్‌లను తేమను పెంచడం మరియు క్రిమిసంహారక సబ్బు లేదా వేప నూనెను ఉపయోగించడం ద్వారా నిర్వహించవచ్చు."
     "మట్టి పురుగులను తొలగించడానికి మొక్కను క్రమం తప్పకుండా నీటితో పిచికారీ చేయండి మరియు జనాభాను నియంత్రించడానికి దోపిడీ పురుగులను ప్రవేశపెట్టడాన్ని పరిగణించండి."
     ),
        'Tomato___Target_Spot': (
     "టార్గెట్ స్పాట్‌ను క్లోరోథలోనిల్ లేదా రాగి-ఆధారిత ఉత్పత్తుల వంటి శిలీంద్రనాశకాలతో చికిత్స చేయవచ్చు."
     "సరైన అంతరం ద్వారా గాలి ప్రసరణను మెరుగుపరచండి, సోకిన ప్రాంతాలను కత్తిరించండి మరియు ఓవర్ హెడ్ నీటిపారుదలని నివారించండి."
     ),
       'Tomato___Tomato_Yellow_Leaf_Curl_Virus':  (
     "వైరస్‌కు చికిత్స లేదు. వ్యాప్తి చెందకుండా నిరోధించడానికి సోకిన మొక్కలను వెంటనే తొలగించండి."
     "ఇమిడాక్లోప్రిడ్ లేదా వేపనూనె వంటి క్రిమిసంహారకాలను ఉపయోగించి క్రిమి వాహకమైన తెల్లదోమలను నియంత్రించండి."
     "సాధ్యమైన చోట వ్యాధి-నిరోధక రకాలను ఉపయోగించండి."
     ),
     'Tomato___Tomato_mosaic_virus': (
     "ఈ వైరస్‌కు చికిత్స లేదు. సోకిన మొక్కలను తొలగించి, సోకిన మొక్కలతో పనిచేసిన తర్వాత సాధనాలను శుభ్రపరచండి."
     "వైరస్ పరిచయం ద్వారా సులభంగా వ్యాప్తి చెందుతుంది కాబట్టి, తడిగా ఉన్నప్పుడు మొక్కలను నిర్వహించడం మానుకోండి."
     )
            }

        },
        'Kannada': {
            'title': "ಸಸ್ಯ ರೋಗ ವರ್ಗೀಕರಣ",
            'upload_prompt': "ನಿಮ್ಮ ಸಸ್ಯದ ಎಲೆಯನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
            'choose_image': "ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
            'predicted_class': "ಅನಿವಾರಿತ ವರ್ಗ",
            'confidence': "ಆತ್ಮವಿಶ್ವಾಸ",
            'solution' : {
    'Apple___Apple_scab': (
     "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ಒಂದು ಶಿಲೀಂಧ್ರ ರೋಗ. ಕ್ಯಾಪ್ಟನ್ ಅಥವಾ ಮೈಕ್ಲೋಬುಟಾನಿಲ್ ಹೊಂದಿರುವ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಅನ್ವಯಿಸಿ."
     "ಶಿಲೀಂಧ್ರದ ಹರಡುವಿಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಯಾವುದೇ ಬಿದ್ದ ಎಲೆಗಳು ಅಥವಾ ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ."
     "ವಾಯು ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಸರಿಯಾದ ಮರದ ಸಮರುವಿಕೆಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ. ಬೆಳವಣಿಗೆಯ ಋತುವಿನಲ್ಲಿ ನಿಯಮಿತ ಅಪ್ಲಿಕೇಶನ್ಗಳು ನಿರ್ಣಾಯಕವಾಗಿವೆ."
     ),
     'Apple___Black_rot': (
     "ಸೋಂಕಿತ ಶಾಖೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದರ ಮೂಲಕ ಕಪ್ಪು ಕೊಳೆತವನ್ನು ನಿಯಂತ್ರಿಸಬಹುದು. ತಾಮ್ರ-ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕಗಳು ಅಥವಾ ದ್ರವ ಸುಣ್ಣದ ಗಂಧಕವನ್ನು ಅನ್ವಯಿಸಿ."
     "ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಮತ್ತು ಜನದಟ್ಟಣೆಯನ್ನು ತಪ್ಪಿಸಲು ಮರಗಳನ್ನು ಕತ್ತರಿಸು. ಬೆಳವಣಿಗೆಯ ಋತುವಿನಲ್ಲಿ, ಕ್ಯಾಪ್ಟನ್ ಅಥವಾ ಥಿಯೋಫನೇಟ್-ಮೀಥೈಲ್ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಣೆಗಳನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ."
     ),
     'Apple___Cedar_apple_rust': (
     "ದಳಗಳ ಪತನದ ಸಮಯದಲ್ಲಿ ಮೈಕ್ಲೋಬುಟಾನಿಲ್ ಅಥವಾ ಪ್ರೊಪಿಕೊನಜೋಲ್ ಹೊಂದಿರುವ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಅನ್ವಯಿಸುವ ಮೂಲಕ ಮತ್ತು ಪ್ರತಿ 7-10 ದಿನಗಳಿಗೊಮ್ಮೆ ಪುನರಾವರ್ತಿಸುವ ಮೂಲಕ ಸೀಡರ್ ಸೇಬಿನ ತುಕ್ಕು ಉತ್ತಮವಾಗಿ ನಿರ್ವಹಿಸಲ್ಪಡುತ್ತದೆ."
     "ಸಮೀಪದ ಯಾವುದೇ ದೇವದಾರು ಮರಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಏಕೆಂದರೆ ಅವು ರೋಗಕ್ಕೆ ಪರ್ಯಾಯ ಸಂಕುಲಗಳಾಗಿವೆ. ಹೊಸ ಮರಗಳನ್ನು ನೆಡುವಾಗ ನಿರೋಧಕ ಸೇಬು ಪ್ರಭೇದಗಳನ್ನು ಬಳಸಿ."
     ),
     'Cherry_(including_sour)_Powdery_mildew': (
     "ಸೂಕ್ಷ್ಮ ಶಿಲೀಂಧ್ರಕ್ಕೆ, ಸಲ್ಫರ್ ಅಥವಾ ಪೊಟ್ಯಾಸಿಯಮ್ ಬೈಕಾರ್ಬನೇಟ್ ಶಿಲೀಂಧ್ರನಾಶಕಗಳು ಪರಿಣಾಮಕಾರಿ."
     "ಸಸ್ಯದ ಸುತ್ತ ಸರಿಯಾದ ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ ಮತ್ತು ಅತಿಯಾದ ಸಾರಜನಕ ಫಲೀಕರಣವನ್ನು ತಪ್ಪಿಸಿ, ಇದು ಮೃದುವಾದ, ಒಳಗಾಗುವ ಬೆಳವಣಿಗೆಯನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ."
     "ಬಾಧಿತ ಎಲೆಗಳು ಮತ್ತು ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸು, ಮತ್ತು ಓವರ್ಹೆಡ್ನಿಂದ ಬದಲಾಗಿ ಸಸ್ಯದ ಬುಡದಲ್ಲಿ ನೀರು."
     ),
     'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot': (
     "ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್ ಅನ್ನು ಸ್ಟ್ರೋಬಿಲುರಿನ್‌ಗಳಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು (ಉದಾ., ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್) ಅಥವಾ ಟ್ರೈಜೋಲ್‌ಗಳನ್ನು (ಉದಾ., ಪ್ರೊಪಿಕೊನಜೋಲ್) ಬಳಸಿಕೊಂಡು ನಿರ್ವಹಿಸಬಹುದು."
     "ರೋಗಕಾರಕ ಸಾಗಣೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಬೆಳೆಗಳ ಉಳಿಕೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬೆಳೆಗಳನ್ನು ತಿರುಗಿಸಿ. ನಿರೋಧಕ ಮಿಶ್ರತಳಿಗಳನ್ನು ನೆಡುವುದನ್ನು ಸಹ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ."
     ),
     'Corn_(maize)_Common_rust': (
     "ಸಾಮಾನ್ಯ ತುಕ್ಕುಗಳನ್ನು ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ ಕ್ಲೋರೋಥಲೋನಿಲ್ ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳಿಂದ ನಿಯಂತ್ರಿಸಬಹುದು."
     "ಸರಿಯಾದ ಬೆಳೆ ತಿರುಗುವಿಕೆಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ ಮತ್ತು ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಹೆಚ್ಚಿನ ಸಸ್ಯ ಸಾಂದ್ರತೆಯನ್ನು ತಪ್ಪಿಸಿ."
     "ತುಕ್ಕು-ನಿರೋಧಕ ಮಿಶ್ರತಳಿಗಳನ್ನು ಬಳಸುವುದು ಅತ್ಯಂತ ಪರಿಣಾಮಕಾರಿ ತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ."
     ),
     'Corn_(maize)_Northern_Leaf_Blight': (
     "ಉತ್ತರ ಎಲೆ ರೋಗವನ್ನು ಸ್ಟ್ರೋಬಿಲುರಿನ್‌ಗಳು (ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್) ಮತ್ತು ಟ್ರೈಜೋಲ್‌ಗಳು (ಟೆಬುಕೊನಜೋಲ್) ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳಿಂದ ನಿಯಂತ್ರಿಸಬಹುದು."
     "ಸಸ್ಯ ನಿರೋಧಕ ಮಿಶ್ರತಳಿಗಳು ಮತ್ತು ರೋಗದ ಚಕ್ರವನ್ನು ಮುರಿಯಲು ಬೆಳೆ ತಿರುಗುವಿಕೆಯನ್ನು ಬಳಸಿ."
     "ಹೊಲಗಳು ಅತಿಯಾಗಿ ನೀರಾವರಿ ಮಾಡದಂತೆ ನೋಡಿಕೊಳ್ಳಿ, ಏಕೆಂದರೆ ತೇವಾಂಶವು ರೋಗದ ತೀವ್ರತೆಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ."
     ),
     'Grape___Black_rot': (
     "ದ್ರಾಕ್ಷಿಯಲ್ಲಿನ ಕಪ್ಪು ಕೊಳೆತಕ್ಕೆ ಕ್ಯಾಪ್ಟನ್, ಮೈಕ್ಲೋಬುಟಾನಿಲ್, ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ ಸೇರಿದಂತೆ ಸ್ಥಿರವಾದ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಪ್ಲಿಕೇಶನ್ ಅಗತ್ಯವಿರುತ್ತದೆ."
     "ಸೋಂಕಿತ ಎಲೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತ್ವರಿತವಾಗಿ ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಹೆಚ್ಚಿಸಲು ಸರಿಯಾದ ಸಮರುವಿಕೆಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
     "ಬೆಳೆಯುವ ಋತುವಿನ ಆರಂಭದಲ್ಲಿ ಶಿಲೀಂಧ್ರನಾಶಕ ಚಿಕಿತ್ಸೆಯನ್ನು ಪ್ರಾರಂಭಿಸಿ ಮತ್ತು ಋತುವಿನ ಉದ್ದಕ್ಕೂ, ವಿಶೇಷವಾಗಿ ಆರ್ದ್ರ ಅವಧಿಗಳಲ್ಲಿ ಮುಂದುವರೆಯಿರಿ."
     ),
     'Grape__Esca(Black_Measles)': (
     "ದುರದೃಷ್ಟವಶಾತ್, ಕಪ್ಪು ದಡಾರಕ್ಕೆ ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಇಲ್ಲ. ಸರಿಯಾದ ನೀರಾವರಿ ಮತ್ತು ಬಳ್ಳಿಗೆ ಗಾಯಗಳನ್ನು ತಪ್ಪಿಸುವ ಮೂಲಕ ತಡೆಗಟ್ಟುವಿಕೆಗೆ ಗಮನ ಕೊಡಿ, ಅದು ಶಿಲೀಂಧ್ರವನ್ನು ಒಳಗೆ ಬಿಡಬಹುದು."
     "ಥಿಯೋಫನೇಟ್-ಮೀಥೈಲ್ ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ರಕ್ಷಣಾತ್ಮಕ ಚಿಕಿತ್ಸೆಯಾಗಿ ಬಳಸುವುದನ್ನು ನೀವು ಪರಿಗಣಿಸಬಹುದು, ಆದರೂ ಇದು ಈಗಾಗಲೇ ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ಗುಣಪಡಿಸುವುದಿಲ್ಲ."
     "ರೋಗ ಹರಡುವುದನ್ನು ತಡೆಗಟ್ಟಲು ತೀವ್ರವಾಗಿ ಸೋಂಕಿತ ಬಳ್ಳಿಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ."
     ),
     'Grape___Leaf_blight(Isariopsis_Leaf_Spot)': (
     "ಎಲೆ ರೋಗವನ್ನು ತಾಮ್ರ-ಆಧಾರಿತ ಸ್ಪ್ರೇಗಳು ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳೊಂದಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಬಹುದು."
     "ವಾಯು ಪ್ರಸರಣವನ್ನು ಅನುಮತಿಸಲು ನಿಯಮಿತ ಸಮರುವಿಕೆಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ ಮತ್ತು ಯಾವುದೇ ಸೋಂಕಿತ ಎಲೆಗಳು ಅಥವಾ ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
     "ಓವರ್ಹೆಡ್ ನೀರಾವರಿ ತಪ್ಪಿಸಿ ಮತ್ತು ಎಲೆಗಳ ಮೇಲಿನ ತೇವಾಂಶವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಹನಿ ನೀರಾವರಿ ಆಯ್ಕೆ ಮಾಡಿ."
     ),
     'Orange___Haunglongbing(Citrus_greening)': (
     "ಸಿಟ್ರಸ್ ಗ್ರೀನಿಂಗ್ ಎನ್ನುವುದು ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಇಲ್ಲದ ಗಂಭೀರ ಕಾಯಿಲೆಯಾಗಿದೆ. ಹರಡುವುದನ್ನು ತಡೆಯಲು ಸೋಂಕಿತ ಮರಗಳನ್ನು ತೆಗೆದುಹಾಕಬೇಕು ಮತ್ತು ನಾಶಪಡಿಸಬೇಕು."
     "ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಅಥವಾ ಥಿಯಾಮೆಥಾಕ್ಸಾಮ್‌ನಂತಹ ಕೀಟನಾಶಕಗಳೊಂದಿಗೆ ರೋಗದ ವಾಹಕವಾದ ಏಷ್ಯನ್ ಸಿಟ್ರಸ್ ಸೈಲಿಡ್ ಅನ್ನು ನಿಯಂತ್ರಿಸಿ."
     "ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮತ್ತು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆಯುವುದು ಸೇರಿದಂತೆ ಸರಿಯಾದ ಹಣ್ಣಿನ ನಿರ್ವಹಣೆ ಅಭ್ಯಾಸಗಳನ್ನು ಅಳವಡಿಸಿ."
     ),
     'Peach___Bacterial_spot': (
     "ತಾಮ್ರ-ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕಗಳು ಅಥವಾ ಆಕ್ಸಿಟೆಟ್ರಾಸೈಕ್ಲಿನ್ ಸ್ಪ್ರೇಗಳನ್ನು ಬಳಸಿಕೊಂಡು ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸ್ಪಾಟ್ ಅನ್ನು ನಿಯಂತ್ರಿಸಬಹುದು."
     "ಸೋಂಕಿತ ಎಲೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ. ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ, ತೇವಾಂಶವು ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಹರಡುವಿಕೆಯನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ."
     "ಹೊಸ ಪೀಚ್ ಮರಗಳನ್ನು ನೆಡುವಾಗ ನಿರೋಧಕ ಪ್ರಭೇದಗಳನ್ನು ಆರಿಸಿ ಮತ್ತು ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಮರಗಳನ್ನು ಕತ್ತರಿಸು."
     ),
     'Pepper_bell___Bacterial_spot': (
     "ತಾಮ್ರ-ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಯಾನಾಶಕಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಅನ್ವಯಿಸುವ ಮೂಲಕ ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸ್ಥಳವನ್ನು ನಿಯಂತ್ರಿಸಿ."
     "ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಸರಿಯಾದ ಬೆಳೆ ಸರದಿಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ. ಮತ್ತಷ್ಟು ಹರಡುವುದನ್ನು ತಡೆಗಟ್ಟಲು ಸೋಂಕಿತ ಎಲೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತ್ವರಿತವಾಗಿ ತೆಗೆದುಹಾಕಿ."
     ),
    'Potato___Early_blight': (
     "ಮುಂಚಿನ ರೋಗಕ್ಕೆ, ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಅನ್ವಯಿಸಿ."
     "ಬೆಳೆಗಳನ್ನು ತಿರುಗಿಸಿ, ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ತೇವಾಂಶವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಸಸ್ಯಗಳ ನಡುವೆ ಸರಿಯಾದ ಅಂತರವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
     "ರೋಗದ ಹರಡುವಿಕೆಯನ್ನು ಮಿತಿಗೊಳಿಸಲು ಸೋಂಕಿತ ಸಸ್ಯದ ಅವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
     ),
     'Potato___Late_blight': (
     "ತಾಮ್ರದ ಆಧಾರದ ಶಿಲೀಂಧ್ರನಾಶಕಗಳೊಂದಿಗೆ ತಡಬ್ಲೈಟ್ ಅನ್ನು ನಿರ್ವಹಿಸಬಹುದು."
     "ರೋಗಕ್ಕೆ ಬೆಳೆ ತಿರುಗುವಿಕೆ ಮತ್ತು ನಿರೋಧಕ ತಳಿ ಆಯ್ಕೆಗಳು ಪ್ರಮುಖ ತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳು."
     "ತಡಬ್ಲೈಟ್ ವಾತಾವರಣವು ತೇವಾಂಶವನ್ನು ಪ್ರೋತ್ಸಾಹಿಸುತ್ತದೆ, ಆದ್ದರಿಂದ ನೀರಾವರಿ ನಿಯಂತ್ರಣೆ ಕೀಲಿಯಾಗಿದೆ."
     ),
    'Squash___Powdery_mildew': (
     "ಸೂಕ್ಷ್ಮ ಶಿಲೀಂಧ್ರಕ್ಕೆ, ಸಲ್ಫರ್, ಬೇವಿನ ಎಣ್ಣೆ ಅಥವಾ ಪೊಟ್ಯಾಸಿಯಮ್ ಬೈಕಾರ್ಬನೇಟ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಅನ್ವಯಿಸಿ."
     "ತೀವ್ರವಾಗಿ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಿ ಮತ್ತು ಎಲೆಗಳನ್ನು ತೇವಗೊಳಿಸುವುದನ್ನು ತಪ್ಪಿಸಲು ಬುಡದಲ್ಲಿರುವ ಸಸ್ಯಗಳಿಗೆ ನೀರು ಹಾಕಿ."
     "ಕಿಕ್ಕಿರಿದ ಸಸ್ಯಗಳನ್ನು ತಪ್ಪಿಸಿ, ಇದು ಶಿಲೀಂಧ್ರದ ಬೆಳವಣಿಗೆಯನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ."
     ),
     'Strawberry___Leaf_scorch': (
     "ಲೀಫ್ ಸ್ಕಾರ್ಚ್ ಅನ್ನು ಕ್ಯಾಪ್ಟನ್ ಅಥವಾ ತಾಮ್ರ-ಆಧಾರಿತ ಸ್ಪ್ರೇಗಳಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳೊಂದಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಬಹುದು."
     "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಸಸ್ಯಗಳ ನಡುವೆ ಸರಿಯಾದ ಅಂತರವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ. ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ."
     ),
     'Tomato___Bacterial_spot': (
     "ತಾಮ್ರ-ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಯಾನಾಶಕಗಳು ಅಥವಾ ಸ್ಟ್ರೆಪ್ಟೊಮೈಸಿನ್ ಅನ್ನು ಬಳಸಿಕೊಂಡು ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸ್ಪಾಟ್ ಅನ್ನು ನಿಯಂತ್ರಿಸಬಹುದು."
     "ಸೋಂಕಿತ ಎಲೆಗಳು ಮತ್ತು ಹಣ್ಣುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬ್ಯಾಕ್ಟೀರಿಯಾವನ್ನು ಹರಡುವುದನ್ನು ತಡೆಯಲು ಆರ್ದ್ರ ಸಸ್ಯಗಳೊಂದಿಗೆ ಕೆಲಸ ಮಾಡುವುದನ್ನು ತಪ್ಪಿಸಿ."
     ),
     'Tomato___Early_blight': (
     "ಆರಂಭಿಕ ರೋಗವನ್ನು ಕ್ಲೋರೋಥಲೋನಿಲ್, ತಾಮ್ರ, ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳಿಂದ ನಿರ್ವಹಿಸಬಹುದು."
     "ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಲು ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸು, ಎಲೆಗಳ ಮೇಲೆ ಮಣ್ಣನ್ನು ಸ್ಪ್ಲಾಶ್ ಮಾಡುವುದನ್ನು ತಡೆಯಲು ಮಲ್ಚ್ ಮತ್ತು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ."
     ),
     'Tomato___Late_blight': (
     "ತಾಮ್ರ-ಆಧಾರಿತ ಉತ್ಪನ್ನಗಳು ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಬಳಸಿಕೊಂಡು ತಡವಾದ ರೋಗವನ್ನು ಚಿಕಿತ್ಸೆ ಮಾಡಬಹುದು."
     "ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ ಮತ್ತು ಓವರ್ಹೆಡ್ ನೀರಾವರಿ ತಪ್ಪಿಸಿ. ಸಾಧ್ಯವಾದರೆ ರೋಗ-ನಿರೋಧಕ ಪ್ರಭೇದಗಳನ್ನು ಬಳಸಿ."
     ),
     'Tomato___Leaf_mold': (
     "ಲೀಫ್ ಅಚ್ಚನ್ನು ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್‌ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳಿಂದ ನಿಯಂತ್ರಿಸಲಾಗುತ್ತದೆ."
     "ಉತ್ತಮ ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ, ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಹೆಚ್ಚು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸು."
     ),
     'Tomato___Septoria_leaf_spot': (
     "ಸೆಪ್ಟೋರಿಯಾ ಲೀಫ್ ಸ್ಪಾಟ್ ಅನ್ನು ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ ಕ್ಲೋರೋಥಲೋನಿಲ್ ನಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳೊಂದಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಬಹುದು."
     "ಮಣ್ಣಿನಿಂದ ಬೀಜಕಗಳ ಹರಡುವಿಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಪೀಡಿತ ಪ್ರದೇಶಗಳು ಮತ್ತು ಮಲ್ಚ್ ಸಸ್ಯಗಳನ್ನು ಕತ್ತರಿಸು."
     ),
     'Tomato___Spider_mites Two-spotted_spider_mite': (
     "ಜೇಡ ಹುಳಗಳನ್ನು ತೇವಾಂಶವನ್ನು ಹೆಚ್ಚಿಸುವ ಮೂಲಕ ಮತ್ತು ಕೀಟನಾಶಕ ಸೋಪ್ ಅಥವಾ ಬೇವಿನ ಎಣ್ಣೆಯನ್ನು ಅನ್ವಯಿಸುವ ಮೂಲಕ ನಿರ್ವಹಿಸಬಹುದು."
     "ಹುಳಗಳನ್ನು ಹೊರಹಾಕಲು ನಿಯಮಿತವಾಗಿ ಸಸ್ಯವನ್ನು ನೀರಿನಿಂದ ಸಿಂಪಡಿಸಿ ಮತ್ತು ಜನಸಂಖ್ಯೆಯನ್ನು ನಿಯಂತ್ರಿಸಲು ಪರಭಕ್ಷಕ ಹುಳಗಳನ್ನು ಪರಿಚಯಿಸುವುದನ್ನು ಪರಿಗಣಿಸಿ."
     ),
     'Tomato___Target_spot': (
     "ಟಾರ್ಗೆಟ್ ಸ್ಪಾಟ್ ಅನ್ನು ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ತಾಮ್ರ-ಆಧಾರಿತ ಉತ್ಪನ್ನಗಳಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳೊಂದಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಬಹುದು."
     "ಸರಿಯಾದ ಅಂತರದಿಂದ ಗಾಳಿಯ ಪ್ರಸರಣವನ್ನು ಸುಧಾರಿಸಿ, ಸೋಂಕಿತ ಪ್ರದೇಶಗಳನ್ನು ಕತ್ತರಿಸು ಮತ್ತು ಓವರ್ಹೆಡ್ ನೀರಾವರಿ ತಪ್ಪಿಸಿ."
     ),
     'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
     "ವೈರಸ್‌ಗೆ ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಇಲ್ಲ. ಹರಡುವುದನ್ನು ತಡೆಯಲು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದುಹಾಕಿ."
     "ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಅಥವಾ ಬೇವಿನ ಎಣ್ಣೆಯಂತಹ ಕೀಟನಾಶಕಗಳನ್ನು ಬಳಸಿಕೊಂಡು ಕೀಟ ವಾಹಕವಾದ ಬಿಳಿ ನೊಣಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ."
     "ಸಾಧ್ಯವಾದಲ್ಲಿ ರೋಗ-ನಿರೋಧಕ ಪ್ರಭೇದಗಳನ್ನು ಬಳಸಿ."
     ),
     'Tomato___Tomato_mosaic_virus': (
     "ಈ ವೈರಸ್‌ಗೆ ಯಾವುದೇ ಚಿಕಿತ್ಸೆ ಇಲ್ಲ. ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಸೋಂಕಿತ ಸಸ್ಯಗಳೊಂದಿಗೆ ಕೆಲಸ ಮಾಡಿದ ನಂತರ ಉಪಕರಣಗಳನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ."
     "ಒದ್ದೆಯಾದಾಗ ಸಸ್ಯಗಳನ್ನು ನಿರ್ವಹಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ, ಏಕೆಂದರೆ ವೈರಸ್ ಸುಲಭವಾಗಿ ಸಂಪರ್ಕದ ಮೂಲಕ ಹರಡುತ್ತದೆ."
     )
            }
        },
        'Malayalam': {
            'title': "പ്ലാൻ്റ് ഡിസീസ് ക്ലാസിഫയർ",
            'upload_prompt': "സസ്യ ഇലയുടെ ചിത്രം അപ്ലോഡ് ചെയ്യുക.",
            'choose_image': "ചിത്രം തിരഞ്ഞെടുക്കുക...",
            'predicted_class': "കണക്കാക്കിയ ക്ലാസ്",
            'confidence': "വിശ്വാസം",
            'solution' : {
    'Apple___Apple_scab': (
     "ആപ്പിൾ ചുണങ്ങു ഒരു കുമിൾ രോഗമാണ്. ക്യാപ്റ്റാൻ അല്ലെങ്കിൽ മൈക്ലോബുട്ടാനിൽ അടങ്ങിയ കുമിൾനാശിനികൾ പ്രയോഗിക്കുക."
     "ഫംഗസിൻ്റെ വ്യാപനം കുറയ്ക്കുന്നതിന് വീണ ഇലകളോ പഴങ്ങളോ നീക്കം ചെയ്ത് നശിപ്പിക്കുക."
     "വായുസഞ്ചാരം മെച്ചപ്പെടുത്താൻ ശരിയായ വൃക്ഷം അരിവാൾ ഉറപ്പാക്കുക. വളരുന്ന സീസണിൽ പതിവ് പ്രയോഗങ്ങൾ നിർണായകമാണ്."
    ),
    'Apple___Black_rot': (
     "രോഗബാധിതമായ ശാഖകളും പഴങ്ങളും നീക്കം ചെയ്യുന്നതിലൂടെ കറുത്ത ചെംചീയൽ നിയന്ത്രിക്കാം. ചെമ്പ് അടിസ്ഥാനമാക്കിയുള്ള കുമിൾനാശിനികളോ ദ്രാവക കുമ്മായം സൾഫറോ പ്രയോഗിക്കുക."
     "വായു സഞ്ചാരം മെച്ചപ്പെടുത്താനും തിരക്ക് ഒഴിവാക്കാനും മരങ്ങൾ വെട്ടിമാറ്റുക. വളരുന്ന സീസണിൽ ക്യാപ്റ്റൻ അല്ലെങ്കിൽ തയോഫാനേറ്റ്-മീഥൈൽ പോലുള്ള കുമിൾനാശിനി സ്പ്രേകൾ ശുപാർശ ചെയ്യുന്നു."
    ),
    'Apple___Cedar_apple_rust': (
     "ദളങ്ങൾ വീഴുമ്പോൾ മൈക്ലോബുട്ടാനിൽ അല്ലെങ്കിൽ പ്രൊപ്പികോണസോൾ അടങ്ങിയ കുമിൾനാശിനികൾ പ്രയോഗിച്ച് ഓരോ 7-10 ദിവസം കൂടുമ്പോഴും ദേവദാരു ആപ്പിൾ തുരുമ്പ് നന്നായി കൈകാര്യം ചെയ്യുന്നു."
     "അടുത്തുള്ള ദേവദാരു മരങ്ങൾ നീക്കം ചെയ്യുക, കാരണം അവ രോഗത്തിനുള്ള ഇതര ആത്മാർത്ഥരാണ്. പുതിയ മരങ്ങൾ നടുമ്പോൾ പ്രതിരോധശേഷിയുള്ള ആപ്പിൾ ഇനങ്ങൾ ഉപയോഗിക്കുക."
    ),
    'Cherry_(including_sour)__Powdery_mildew': (
     "ടിന്നിന് വിഷമഞ്ഞു, സൾഫർ അല്ലെങ്കിൽ പൊട്ടാസ്യം ബൈകാർബണേറ്റ് കുമിൾനാശിനികൾ ഫലപ്രദമാണ്."
     "സസ്യത്തിന് ചുറ്റും ശരിയായ വായുസഞ്ചാരം ഉറപ്പാക്കുകയും അമിതമായ നൈട്രജൻ വളപ്രയോഗം ഒഴിവാക്കുകയും ചെയ്യുക, ഇത് മൃദുവായതും സാധ്യതയുള്ളതുമായ വളർച്ചയെ പ്രോത്സാഹിപ്പിക്കും."
     "ബാധിതമായ ഇലകളും ശാഖകളും വെട്ടിമാറ്റുക, കൂടാതെ ചെടിയുടെ ചുവട്ടിൽ വെള്ളം ഒഴിക്കുക."
    ),
    'Corn_(Maize)__Cercospora_leaf_spot Gray_leaf_spot': (
     "സ്ട്രോബിലൂരിൻസ് (ഉദാ. അസോക്സിസ്ട്രോബിൻ) അല്ലെങ്കിൽ ട്രയാസോൾ (ഉദാ. പ്രൊപികോണസോൾ) പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ചാരനിറത്തിലുള്ള ഇലപ്പുള്ളി നിയന്ത്രിക്കാം."
     "വിളകളുടെ അവശിഷ്ടങ്ങൾ നീക്കം ചെയ്യുക, രോഗകാരികളുടെ വ്യാപനം കുറയ്ക്കുന്നതിന് വിളകൾ തിരിക്കുക. പ്രതിരോധശേഷിയുള്ള സങ്കരയിനങ്ങൾ നടുന്നതും ശുപാർശ ചെയ്യുന്നു."
    ),
    'Corn__Common_rust': (
     "സാധാരണ തുരുമ്പിനെ മാങ്കോസെബ് അല്ലെങ്കിൽ ക്ലോറോത്തലോനിൽ പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് നിയന്ത്രിക്കാം."
     "ശരിയായ വിള ഭ്രമണം ഉറപ്പാക്കുക, വായു സഞ്ചാരം മെച്ചപ്പെടുത്തുന്നതിന് ഉയർന്ന സസ്യ സാന്ദ്രത ഒഴിവാക്കുക."
     "തുരുമ്പ്-പ്രതിരോധശേഷിയുള്ള സങ്കരയിനം ഉപയോഗിക്കുന്നത് ഏറ്റവും ഫലപ്രദമായ പ്രതിരോധ നടപടികളിൽ ഒന്നാണ്."
    ),
    'Corn__Northern_Leaf_Blight': (
     "സ്‌ട്രോബിലൂറിൻസ് (അസോക്സിസ്ട്രോബിൻ), ട്രയാസോൾസ് (ടെബുകോണസോൾ) തുടങ്ങിയ കുമിൾനാശിനികൾ ഉപയോഗിച്ച് വടക്കൻ ഇലച്ചെടിയെ നിയന്ത്രിക്കാം."
     "പ്രതിരോധശേഷിയുള്ള സങ്കരയിനങ്ങളെ നട്ടുപിടിപ്പിക്കുക, രോഗചക്രം തകർക്കാൻ വിള ഭ്രമണം ഉപയോഗിക്കുക."
     "വയലുകളിൽ കൂടുതൽ ജലസേചനം നടത്തുന്നില്ലെന്ന് ഉറപ്പാക്കുക, കാരണം ഈർപ്പം രോഗത്തിൻറെ തീവ്രത വർദ്ധിപ്പിക്കും."
    ),
    'Grape__Black_rot': (
     "മുന്തിരിയിലെ കറുത്ത ചെംചീയലിന് ക്യാപ്റ്റാൻ, മൈക്ലോബുട്ടാനിൽ അല്ലെങ്കിൽ മാങ്കോസെബ് ഉൾപ്പെടെയുള്ള സ്ഥിരമായ കുമിൾനാശിനി പ്രയോഗം ആവശ്യമാണ്."
     "ബാധയേറ്റ ഇലകളും പഴങ്ങളും ഉടനടി നീക്കം ചെയ്യുക, വായു സഞ്ചാരം വർദ്ധിപ്പിക്കുന്നതിന് ശരിയായ അരിവാൾ ഉറപ്പാക്കുക."
     "വളരുന്ന സീസണിൻ്റെ തുടക്കത്തിൽ കുമിൾനാശിനി ചികിത്സ ആരംഭിക്കുക, സീസണിലുടനീളം, പ്രത്യേകിച്ച് ആർദ്ര കാലഘട്ടങ്ങളിൽ തുടരുക."
    ),
    'Grape__Esca(Black_Measles)': (
     "നിർഭാഗ്യവശാൽ, ബ്ലാക്ക് മീസിൽസിന് ചികിത്സയില്ല. ശരിയായ ജലസേചനം ഉറപ്പാക്കുകയും മുന്തിരിവള്ളിയുടെ പരിക്കുകൾ ഒഴിവാക്കുകയും ചെയ്തുകൊണ്ട് പ്രതിരോധത്തിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക, ഇത് ഫംഗസിനെ അകത്തേക്ക് കടത്തിവിടും."
     "തയോഫാനേറ്റ്-മീഥൈൽ പോലുള്ള കുമിൾനാശിനികൾ ഒരു സംരക്ഷിത ചികിത്സയായി ഉപയോഗിക്കുന്നത് നിങ്ങൾക്ക് പരിഗണിക്കാം, എന്നിരുന്നാലും ഇത് ഇതിനകം രോഗം ബാധിച്ച ചെടികളെ സുഖപ്പെടുത്തില്ല."
     "രോഗം പടരാതിരിക്കാൻ ഗുരുതരമായി ബാധിച്ച വള്ളികൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക."
    ),
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': (
     "ചെമ്പ് അധിഷ്ഠിത സ്പ്രേകൾ അല്ലെങ്കിൽ മാങ്കോസെബ് പോലെയുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ഇല വാട്ടം ചികിത്സിക്കാം."
     "വായു സഞ്ചാരം അനുവദിക്കുന്നതിന് പതിവായി അരിവാൾകൊണ്ടുവരുന്നത് ഉറപ്പാക്കുക, രോഗബാധയുള്ള ഇലകളോ പഴങ്ങളോ നീക്കം ചെയ്യുക."
     "ഓവർഹെഡ് ജലസേചനം ഒഴിവാക്കുക, ഇലകളിലെ ഈർപ്പം കുറയ്ക്കുന്നതിന് ഡ്രിപ്പ് ഇറിഗേഷൻ തിരഞ്ഞെടുക്കുക."
    ),
    'Orange__Haunglongbing(Citrus_greening)': (
     "സിട്രസ് ഗ്രീനിംഗ് ഒരു ഗുരുതരമായ രോഗമാണ്, ചികിത്സയില്ല. രോഗം പടരാതിരിക്കാൻ രോഗം ബാധിച്ച മരങ്ങൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും വേണം."
     "ഇമിഡാക്ലോപ്രിഡ് അല്ലെങ്കിൽ തയാമെത്തോക്സാം പോലുള്ള കീടനാശിനികൾ ഉപയോഗിച്ച് രോഗത്തിൻ്റെ പ്രാണികളുടെ വാഹകനായ ഏഷ്യൻ സിട്രസ് സൈലിഡിനെ നിയന്ത്രിക്കുക."
     "നിരന്തര നിരീക്ഷണവും രോഗബാധയുള്ള ചെടികൾ നീക്കം ചെയ്യുന്നതും ഉൾപ്പെടെ ശരിയായ തോട്ട പരിപാലന രീതികൾ നടപ്പിലാക്കുക."
    ),
    'Peach___Bacterial_spot': (
     "ചെമ്പ് അധിഷ്ഠിത കുമിൾനാശിനികൾ അല്ലെങ്കിൽ ഓക്സിടെട്രാസൈക്ലിൻ സ്പ്രേകൾ ഉപയോഗിച്ച് ബാക്ടീരിയൽ സ്പോട്ട് നിയന്ത്രിക്കാം."
     "രോഗബാധിതമായ ഇലകളും പഴങ്ങളും നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക. ഈർപ്പം ബാക്ടീരിയയുടെ വ്യാപനത്തെ പ്രോത്സാഹിപ്പിക്കുന്നതിനാൽ മുകളിലൂടെ നനവ് ഒഴിവാക്കുക."
     "പുതിയ പീച്ച് മരങ്ങൾ നട്ടുപിടിപ്പിക്കുമ്പോൾ പ്രതിരോധശേഷിയുള്ള ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക, വായു സഞ്ചാരം മെച്ചപ്പെടുത്താൻ മരങ്ങൾ മുറിക്കുക."
    ),
    'Pepper__Bacterial_spot': (
     "ചെമ്പ് അധിഷ്ഠിത ബാക്ടീരിയനാശിനികൾ പതിവായി പ്രയോഗിച്ച് ബാക്ടീരിയൽ സ്പോട്ട് നിയന്ത്രിക്കുക."
     "ഓവർഹെഡ് നനവ് ഒഴിവാക്കുകയും ശരിയായ വിള ഭ്രമണം ഉറപ്പാക്കുകയും ചെയ്യുക. കൂടുതൽ പടരാതിരിക്കാൻ രോഗം ബാധിച്ച ഇലകളും പഴങ്ങളും ഉടനടി നീക്കം ചെയ്യുക."
    ),
    'Potato___Early_blight': (
     "നേരത്തെ വരൾച്ചയ്ക്ക്, ക്ലോറോത്തലോനിൽ അല്ലെങ്കിൽ മാങ്കോസെബ് പോലുള്ള കുമിൾനാശിനികൾ പ്രയോഗിക്കുക."
     "വിളകൾ തിരിക്കുക, ഓവർഹെഡ് നനവ് ഒഴിവാക്കുക, ഈർപ്പം കുറയ്ക്കുന്നതിന് ചെടികൾക്കിടയിൽ ശരിയായ അകലം ഉറപ്പാക്കുക."
     "രോഗം പടരുന്നത് പരിമിതപ്പെടുത്താൻ രോഗബാധയുള്ള ചെടികളുടെ അവശിഷ്ടങ്ങൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക."
    ),
    'Potato___Late_blight': (
     "വൈകി വരൾച്ച ചെമ്പ് അധിഷ്ഠിത ഉൽപ്പന്നങ്ങൾ പോലുള്ള കുമിൾനാശിനികൾ അല്ലെങ്കിൽ മെഫെനോക്സാം പോലുള്ള വ്യവസ്ഥാപരമായ കുമിൾനാശിനികൾ ഉപയോഗിച്ച് നിയന്ത്രിക്കാം."
     "പടരുന്നത് തടയാൻ രോഗബാധയുള്ള ചെടികളും കിഴങ്ങുകളും നശിപ്പിക്കുക. മുകളിലൂടെ നനവ് ഒഴിവാക്കുകയും ശരിയായ മണ്ണ് ഒഴുകുന്നത് ഉറപ്പാക്കുകയും ചെയ്യുക."
    ),
    'Squash___Powdery_mildew': (
     "പോഡറി മിൽഡ്യൂക്ക് സൾഫർ, വേപ്പെണ്ണ അല്ലെങ്കിൽ പൊട്ടാസ്യം ബൈകാർബണേറ്റ് പോലുള്ള കുമിൾനാശിനികൾ പ്രയോഗിക്കുക."
     "തീവ്രമായി ബാധിച്ച ഇലകൾ നീക്കം ചെയ്യുക, വായുസഞ്ചാരം മെച്ചപ്പെടുത്തുക, സസ്യജാലങ്ങൾ നനയാതിരിക്കാൻ ചുവട്ടിലെ ചെടികൾ നനയ്ക്കുക."
     "തിരക്കേറിയ ചെടികൾ ഒഴിവാക്കുക, ഇത് പൂപ്പലിൻ്റെ വളർച്ചയെ പ്രോത്സാഹിപ്പിക്കുന്നു."
    ),
    'Strawberry___Leaf_scorch': (
     "ഇല പൊള്ളൽ ക്യാപ്റ്റാൻ അല്ലെങ്കിൽ ചെമ്പ് അടിസ്ഥാനമാക്കിയുള്ള സ്പ്രേകൾ പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ചികിത്സിക്കാം."
     "വായുസഞ്ചാരം മെച്ചപ്പെടുത്തുന്നതിനായി രോഗബാധയുള്ള ഇലകൾ നീക്കം ചെയ്യുകയും ചെടികൾക്കിടയിൽ ശരിയായ അകലം ഉറപ്പാക്കുകയും ചെയ്യുക. മുകളിലൂടെ നനവ് ഒഴിവാക്കുക."
    ),
    'Tomato___Bacterial_spot': (
     "ബാക്ടീരിയൽ സ്പോട്ട് ചെമ്പ് അധിഷ്ഠിത ബാക്ടീരിയനാശിനികൾ അല്ലെങ്കിൽ സ്ട്രെപ്റ്റോമൈസിൻ ഉപയോഗിച്ച് നിയന്ത്രിക്കാം."
     "ബാധിച്ച ഇലകളും പഴങ്ങളും നീക്കം ചെയ്യുക, ബാക്ടീരിയ പടരുന്നത് തടയാൻ നനഞ്ഞ ചെടികളുമായി പ്രവർത്തിക്കുന്നത് ഒഴിവാക്കുക."
    ),
    'Tomato___Early_blight': (
     "ആദ്യകാല വരൾച്ച ക്ലോറോത്തലോനിൽ, ചെമ്പ് അല്ലെങ്കിൽ മാങ്കോസെബ് പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് നിയന്ത്രിക്കാം."
     "വായു സഞ്ചാരം മെച്ചപ്പെടുത്താൻ താഴത്തെ ഇലകൾ മുറിക്കുക, ഇലകളിൽ മണ്ണ് തെറിക്കുന്നത് തടയാൻ പുതയിടുക, രോഗം ബാധിച്ച ഇലകൾ നീക്കം ചെയ്യുക."
    ),
    'Tomato___Late_blight': (
     "വൈകി വരൾച്ച ചെമ്പ് അധിഷ്ഠിത ഉൽപ്പന്നങ്ങൾ അല്ലെങ്കിൽ മാങ്കോസെബ് പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ചികിത്സിക്കാം."
     "രോഗബാധിതമായ ചെടികൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക, മുകളിലെ ജലസേചനം ഒഴിവാക്കുക. സാധ്യമാകുന്നിടത്ത് രോഗ പ്രതിരോധശേഷിയുള്ള ഇനങ്ങൾ ഉപയോഗിക്കുക."
    ),
    'Tomato___Leaf_Mold': (
     "ഇല പൂപ്പൽ ക്ലോറോത്തലോനിൽ അല്ലെങ്കിൽ മാങ്കോസെബ് പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ചാണ് നിയന്ത്രിക്കുന്നത്."
     "നല്ല വായുസഞ്ചാരം ഉറപ്പാക്കുക, മുകളിലൂടെ നനവ് ഒഴിവാക്കുക, സാരമായി ബാധിച്ച ഇലകൾ വെട്ടിമാറ്റുക."
    ),
    'Tomato___Septoria_leaf_spot': (
     "സെപ്റ്റോറിയ ഇലപ്പുള്ളി മങ്കോസെബ് അല്ലെങ്കിൽ ക്ലോറോത്തലോനിൽ പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ചികിത്സിക്കാം."
     "മണ്ണിൽ നിന്നുള്ള ബീജങ്ങളുടെ വ്യാപനം കുറയ്ക്കുന്നതിന് ബാധിത പ്രദേശങ്ങളും ചവറുകൾ ചെടികളും വെട്ടിമാറ്റുക."
    ),
    'Tomato___Spider_mites_Two-spotted_spider_mite': (
     "ഈർപ്പം വർദ്ധിപ്പിച്ച് കീടനാശിനി സോപ്പ് അല്ലെങ്കിൽ വേപ്പെണ്ണ പ്രയോഗിച്ച് ചിലന്തി കാശ് നിയന്ത്രിക്കാം."
     "കാഷ് തുരത്താൻ പതിവായി ചെടിയിൽ വെള്ളം തളിക്കുക, ജനസംഖ്യ നിയന്ത്രിക്കാൻ ഇരപിടിക്കുന്ന കാഷ് അവതരിപ്പിക്കുന്നതുപരിഗണിക്കുക."
    ),
    'Tomato___Target_Spot': (
     "ടാർഗെറ്റ് സ്പോട്ട് ക്ലോറോത്തലോനിൽ അല്ലെങ്കിൽ ചെമ്പ് അടിസ്ഥാനമാക്കിയുള്ള ഉൽപ്പന്നങ്ങൾ പോലുള്ള കുമിൾനാശിനികൾ ഉപയോഗിച്ച് ചികിത്സിക്കാം."
     "ശരിയായ അകലത്തിലൂടെ വായു സഞ്ചാരം മെച്ചപ്പെടുത്തുക, രോഗബാധിത പ്രദേശങ്ങൾ വെട്ടിമാറ്റുക, മുകളിലൂടെയുള്ള ജലസേചനം ഒഴിവാക്കുക."
    ),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
     "വൈറസിന് ചികിത്സയില്ല. പടരാതിരിക്കാൻ രോഗം ബാധിച്ച ചെടികൾ ഉടനടി നീക്കം ചെയ്യുക."
     "ഇമിഡാക്ലോപ്രിഡ് അല്ലെങ്കിൽ വേപ്പെണ്ണ പോലുള്ള കീടനാശിനികൾ ഉപയോഗിച്ച് പ്രാണികളുടെ വിതരണം ചെയ്യുന്ന വെള്ളീച്ചകളെ നിയന്ത്രിക്കുക."
     "സാധ്യമാകുന്നിടത്ത് രോഗ പ്രതിരോധ ഇനങ്ങൾ ഉപയോഗിക്കുക."
    ),
    'Tomato___Tomato_mosaic_virus': (
     "ഈ വൈറസിന് ചികിത്സയില്ല. രോഗബാധിതമായ ചെടികൾ നീക്കം ചെയ്യുകയും രോഗബാധിതമായ ചെടികളുമായി പ്രവർത്തിച്ചതിന് ശേഷം ഉപകരണങ്ങൾ അണുവിമുക്തമാക്കുകയും ചെയ്യുക."
     "നനഞ്ഞപ്പോൾ ചെടികൾ കൈകാര്യം ചെയ്യുന്നത് ഒഴിവാക്കുക, കാരണം വൈറസ് സമ്പർക്കത്തിലൂടെ എളുപ്പത്തിൽ പടരാൻ കഴിയും."
    )
            }
        }
    }

    def load_model():
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(class_names)),
            nn.LogSoftmax(dim=1)
        )
        model.load_state_dict(torch.load("best_model_finetuned.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        return model

    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_AUGMENTED, std=STD_AUGMENTED),
        ])
        image = transform(image).unsqueeze(0)
        return image

    model = load_model()

    language = st.selectbox("Select Language", ['English', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Malayalam'])

    texts = translations[language]

    st.title(texts['title'])
    st.write(texts['upload_prompt'])

    uploaded_file = st.file_uploader(texts['choose_image'], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class = class_names[predicted.item()]

        st.write(f"{texts['predicted_class']}: **{predicted_class}**")

        probabilities = torch.softmax(outputs, dim=1)
        confidence = probabilities[0, predicted.item()].item()
        st.write(f"{texts['confidence']}: **{confidence:.4f}**")

        if predicted_class not in ['Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
                                   'Corn_(maize)___healthy', 'Grape___healthy', 'Peach___healthy', 'Pepper,_bell___healthy',
                                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Strawberry___healthy',
                                   'Tomato___healthy']:

            if predicted_class in texts['solution']:
                st.header(f"**Solution for {predicted_class}:**")
                st.write(texts['solution'][predicted_class])
            else:
                st.write("Solution not available for this disease in the selected language.")
        else:
            st.write("Your plant appears to be healthy. Keep up the good care :)!")
import os
API_KEY = os.getenv("API_KEY")
import google.generativeai as genai
from googletrans import Translator

genai.configure(api_key=API_KEY)


def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text, translation.src

def translate_to_original(text, src_lang):
    translator = Translator()
    translation = translator.translate(text, dest=src_lang)
    return translation.text
    

import re

def is_agriculture_related(query):
    agriculture_keywords = [
        "farm", "farming", "agriculture", "plant", "crop", "food", "vegetable", 
        "fruits", "soil", "harvest", "pest", "irrigation", "fertilizer", "seeds", 
        "disease", "yield", "weather", "climate", "rain", "sun", "humidity", 
        "temperature", "drought", "greenhouse", "sustainable farming", "pesticide", 
        "herbicide", "fungicide", "crop rotation", "organic farming", "compost", 
        "mulching", "vermicomposting", "biodiversity", "monoculture", "polyculture", 
        "intercropping", "cover crops", "cash crops", "staple crops", "cereal", 
        "grain", "legume", "pulse", "root vegetable", "tuber", "oilseed", "silage", 
        "weed control", "crop protection", "biological control", "plant nutrition", 
        "soil erosion", "crop water requirement", "soil fertility", "land degradation", 
        "bacterial disease", "fungal disease", "viral disease", "blight", "mosaic virus", 
        "rust", "powdery mildew", "downy mildew", "late blight", "early blight", 
        "leaf spot", "root rot", "stem rot", "fruit rot", "anthracnose", "canker", 
        "wilt", "smut", "scab", "nematode", "aphid", "mealybug", "thrips", "whitefly", 
        "cutworm", "armyworm", "leafhopper", "bollworm", "stem borer", "fruit borer", 
        "grub", "maggot", "mite", "trace elements", "macro nutrients", "micro nutrients", 
        "photosynthesis", "respiration", "transpiration", "chlorophyll", "green manure", 
        "soil structure", "soil moisture", "water retention", "crop stress", "pest outbreak", 
        "farmer income", "soil testing", "seed technology", "crop diseases",
        "apple", "banana", "orange", "grape", "mango", "papaya", "pineapple", 
        "watermelon", "melon", "strawberry", "blueberry", "raspberry", "blackberry", 
        "pear", "peach", "plum", "cherry", "pomegranate", "kiwi", "avocado", 
        "fig", "date", "apricot", "lemon", "lime", "grapefruit", "tangerine", 
        "coconut", "guava", "lychee", "jackfruit", "custard apple", "durian", 
        "dragonfruit", "persimmon", "passionfruit", "sapodilla", "starfruit", 
        "mulberry", "cranberry", "quince", "tomato", "potato", "onion", "carrot", "cabbage", 
        "cauliflower", "broccoli", "spinach", "lettuce", "kale", "cucumber", "zucchini", 
        "eggplant", "bell pepper", "pumpkin", "squash", "radish", "turnip", "sweet potato", 
        "beetroot", "garlic", "ginger", "chili", "green bean", "peas", "okra", 
        "asparagus", "artichoke", "celery", "brussels sprouts", "leek", "shallot", 
        "parsnip", "fennel", "collard greens", "swiss chard", "watercress", "mushroom", 
        "bamboo shoot", "lotus root", "gourd", "bitter melon", "yam", "taro", "daikon", 
        "horseradish", "jicama", "chayote", "rose", "jasmine", "marigold", "hibiscus", 
        "lavender", "sunflower", "daisy", "tulip", "orchid", "lily", "daffodil", 
        "geranium", "chrysanthemum", "violet", "petunia", "zinnia", "begonia", 
        "gladiolus", "freesia", "snapdragon", "lotus", "poppy", "carnation", "pansy", 
        "morning glory", "cosmos", "calendula", "foxglove", "hollyhock", "edelweiss", 
        "bluebell", "dahlia", "gerbera", "black pepper", "turmeric", "cinnamon", 
        "clove", "cardamom", "coriander", "cumin", "fenugreek", "mustard", 
        "star anise", "bay leaf", "saffron", "paprika", "nutmeg", "mace", "oregano", 
        "basil", "thyme", "rosemary", "marjoram", "dill", "tarragon", "fennel seed", 
        "chili pepper", "ginger root", "vanilla", "caraway", "sumac", "lemongrass", 
        "allspice", "citrus greening", "fire blight", "peach leaf curl", "downy mildew", 
        "verticillium wilt", "fusarium wilt", "gray mold", "powdery mildew", 
        "clubroot", "botrytis", "gummosis", "apple scab", "black rot", "leaf curl", 
        "fruit rot", "sooty mold", "mosaic virus", "fruit fly", "codling moth", 
        "root-knot nematode", "aphids", "spider mites", "whiteflies", "thrips", 
        "cabbage looper", "tomato hornworm", "potato beetle", "melon fly"
    ]

    query_lower = query.lower()

    return any(re.search(rf'\b{re.escape(keyword)}s?\b', query_lower) for keyword in agriculture_keywords)
    
if mode == "Chatbot Mode": 
    st.header("Chatbot for Farmers 👒")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("You: ")
    
    if user_input:
        user_input_translated, src_lang = translate_to_english(user_input)
        if is_agriculture_related(user_input_translated):
            st.session_state.messages.append({"role": "user", "content": user_input})
    
            with st.chat_message("user"):
                st.markdown(user_input)
    
            with st.spinner("Translating and generating response..."):
                response = genai.GenerativeModel('gemini-1.5-pro-latest').generate_content(user_input_translated)
                translated_response = translate_to_original(response.text, src_lang)
    
                with st.chat_message("assistant"):
                    st.markdown(translated_response)
    
                st.session_state.messages.append({"role": "assistant", "content": translated_response})
        else:
            with st.chat_message("assistant"):
                st.markdown("I can only assist with topics related to farming, agriculture, plants, crops, and food. Please try again with a relevant query.")
                
elif mode == "Image to Text":
    st.header("Image to Text")

    uploaded_image = st.file_uploader("Upload an image for conversion to text", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        description_type = st.text_area(
            "What should the image description focus on? (e.g., 'summarize what is in the image')")

        if st.button("Convert Image to Text"):
            with st.spinner("Processing image to text..."):
                translated_description, src_lang = translate_to_english(description_type if description_type else "")
                input_data = [f"Convert this image to text. {translated_description}", img]

                response = genai.GenerativeModel('gemini-1.5-pro-latest').generate_content(input_data)
                translated_response = translate_to_original(response.text, src_lang)
                st.write(translated_response)
