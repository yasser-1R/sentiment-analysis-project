import joblib

# === Chargement du modèle et du vectorizer
model_path = "NuSVC_best_model.pkl"
vectorizer_path = "NuSVC_vectorizer.pkl"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    exit()

print("\n🧠 موديل NuSVC لتحليل المشاعر بالعربية")
print("أدخل جملة بالعربية (أو اكتب 'exit' للخروج):\n")

while True:
    sentence = input("📥 جملتك: ").strip()

    if sentence.lower() == "exit":
        print("👋 إلى اللقاء !")
        break

    if not sentence:
        print("⚠️ من فضلك أدخل جملة صالحة.\n")
        continue

    # Vectorisation et prédiction
    try:
        X_input = vectorizer.transform([sentence])
        prediction = model.predict(X_input)[0]
        result = "🟢 إيجابي" if prediction == 1 else "🔴 سلبي"
        print(f"📊 النتيجة: {result}\n")
    except Exception as e:
        print(f"⚠️ خطأ أثناء التنبؤ: {e}\n")
