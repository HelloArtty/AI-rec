import json
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
    
class MealPlanner(ABC):
    def __init__(self):
        self.url = os.getenv("URL")

    def get_data(self):
        """ดึงข้อมูลจาก API"""
        url = self.url + "get_limit"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            print("📥 ข้อมูลถูกดึงมาเรียบร้อยแล้ว!")
            # print(json.dumps(data, indent=4))
            if not isinstance(data, dict):
                raise ValueError("Invalid data format from API")
            return data
        except requests.exceptions.RequestException as err:
            print(f"❌ Error fetching data: {err}")
            return None
        except (json.JSONDecodeError, ValueError) as err:
            print(f"❌ Error processing data: {err}")
            return None

    @staticmethod
    def calculate_total_nutrition(meals):
        """คำนวณสารอาหารรวม"""
        valid_meals = [meal for meal in meals if isinstance(meal, dict) and "nutrition" in meal]
        if not valid_meals:
            return {}
        
        total_nutrition = {key: 0 for key in valid_meals[0]["nutrition"].keys()}
        for meal in valid_meals:
            for key, value in meal["nutrition"].items():
                total_nutrition[key] += value
        return total_nutrition

    @staticmethod
    def is_within_nutrition_limit(meals, nutrition_limit):
        """ตรวจสอบขอบเขตโภชนาการ"""
        total_nutrition = MealPlanner.calculate_total_nutrition(meals)
        return all(total_nutrition.get(key, 0) <= nutrition_limit.get(key, float('inf')) 
                for key in nutrition_limit)

    @abstractmethod
    def process_mealplan(self, *args, **kwargs):
        """Abstract method to be implemented by child classes"""
        pass

    @abstractmethod
    def send_mealplan(self, mealplan):
        """ ส่งแผนมื้ออาหารไปยังเซิร์ฟเวอร์ """
        url = os.getenv("URL") + "mealplan"
        try:
            response = requests.post(url, json=mealplan)
            response.raise_for_status()
            print("✅ แผนมื้ออาหารถูกส่งไปยังเซิร์ฟเวอร์เรียบร้อยแล้ว!")
        except requests.exceptions.RequestException as err:
            print(f"❌ Error sending meal plan: {err}")

class CreateMealPlan(MealPlanner):
    def __init__(self):
        super().__init__()
        

    def find_optimal_clusters(self, food_menus):
        """ค้นหาจำนวนคลัสเตอร์ที่เหมาะสม"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        meal_names = [meal['name'] for meal in food_menus]
        embeddings = model.encode(meal_names)
        
        unique_embeddings = np.unique(embeddings, axis=0)
        max_clusters = min(10, len(unique_embeddings))
        
        distortions = []
        K = range(1, max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            distortions.append(kmeans.inertia_)
        
        if not distortions:
            raise ValueError("No distortions calculated, check the input data.")
        
        optimal_clusters = np.argmin(np.diff(distortions)) + 2 if len(distortions) > 1 else 1
        return optimal_clusters

    def cluster_meals(self, food_menus):
        """จัดกลุ่มอาหาร"""
        num_clusters = self.find_optimal_clusters(food_menus)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        meal_names = [meal['name'] for meal in food_menus]
        embeddings = model.encode(meal_names)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        clustered_meals = {i: [] for i in range(num_clusters)}
        for meal, cluster in zip(food_menus, clusters):
            clustered_meals[cluster].append(meal)
        
        return clustered_meals

    def process_mealplan(self, food_data):
        """สร้างแผนมื้ออาหาร"""
        if not isinstance(food_data, dict) or "food_menus" not in food_data:
            raise ValueError("Invalid food_data format")
        
        food_menus = food_data["food_menus"]
        user_id = food_data.get("user_id")
        days = food_data.get("days")
        # days = 1
        nutrition_limit = food_data.get("nutrition_limit_per_day", {})
        
        clustered_meals = self.cluster_meals(food_menus)
        
        mealplan = {
            "user_id": user_id,
            "mealplans": [],
            # "total_nutrition_per_day": []
        }
        selected_meals = set()
        
        for _ in range(days):
            daily_meals = []
            for cluster in clustered_meals.values():
                random.shuffle(cluster)
                for meal in cluster:
                    if len(daily_meals) >= 3:
                        break
                    if (meal["name"] not in selected_meals and
                        self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit)):
                        daily_meals.append(meal)
                        selected_meals.add(meal["name"])
                if len(daily_meals) >= 3:
                    break
            
            # total_nutrition = self.calculate_total_nutrition(daily_meals)
            mealplan["mealplans"].append(daily_meals if daily_meals else [])
            # mealplan["total_nutrition_per_day"].append(total_nutrition)

        return mealplan

    def send_mealplan(self, mealplan):
        """ส่งแผนมื้ออาหารไปยังเซิร์ฟเวอร์"""
        url = self.url + "mealplan"
        try:
            response = requests.post(url, json=mealplan)
            response.raise_for_status()
            print("✅ แผนมื้ออาหารถูกส่งไปยังเซิร์ฟเวอร์เรียบร้อยแล้ว!")
        except requests.exceptions.RequestException as err:
            print(f"❌ Error sending meal plan: {err}")
            raise HTTPException(status_code=500, detail=str(err))

class UpdateMealPlan:
    def __init__(self):
        self.url = os.getenv("URL")
    
    def get_data(self):
        """ ดึงข้อมูลจาก API """
        url = self.url + "get_limit"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "food_menus" not in data or "nutrition_limit_per_day" not in data:
                print(f"❌ Invalid data format: {data}")
                return None
            
            return data
        except requests.exceptions.RequestException as err:
            print(f"❌ Error fetching data: {err}")
            return None
    
    def get_mealplan(self):
        """ ดึงข้อมูลแผนมื้ออาหารจาก API """
        url = self.url + "get_mealplan"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "mealplans" not in data:
                print(f"❌ Key 'mealplans' not found in response: {data}")
                data["mealplans"] = []
            
            return data
        except requests.exceptions.RequestException as err:
            print(f"❌ Error fetching meal plan: {err}")
            return None
    
    def send_mealplan(self, mealplan):
        """ ส่งแผนมื้ออาหารไปยังเซิร์ฟเวอร์ """
        url = f"{self.url}update_mealplan"
        try:
            response = requests.post(url, json=mealplan)
            response.raise_for_status()
            print("✅ แผนมื้ออาหารถูกส่งไปยังเซิร์ฟเวอร์เรียบร้อยแล้ว!")
        except requests.exceptions.RequestException as err:
            print(f"❌ Error sending meal plan: {err}")
    
    def calculate_total_nutrition(self, meals):
        """ คำนวณสารอาหารรวมจากมื้ออาหารที่ไม่เป็น {} """
        valid_meals = [meal for meal in meals if isinstance(meal, dict) and "nutrition" in meal]
        if not valid_meals:
            return {}
        
        total_nutrition = {key: 0 for key in valid_meals[0]["nutrition"].keys()}
        
        for meal in valid_meals:
            for key, value in meal["nutrition"].items():
                total_nutrition[key] += value
        
        return total_nutrition
    
    def is_within_nutrition_limit(self, meals, nutrition_limit):
        total_nutrition = self.calculate_total_nutrition(meals)
        return all(total_nutrition.get(key, 0) <= nutrition_limit.get(key, float('inf')) for key in nutrition_limit)
    
    def update_mealplan(self, mealplan, food_menus, nutrition_limit):
        total_nutrition_per_day = []

        for daily_meals in mealplan["mealplans"]:
            used_recipes = {meal["recipe_id"] for meal in daily_meals if isinstance(meal, dict) and "recipe_id" in meal}

            for i in range(len(daily_meals)):
                if daily_meals[i] == {}:  # ตรวจสอบว่ามื้ออาหารเป็นค่าว่างหรือไม่
                    available_meals = [meal for meal in food_menus if meal["recipe_id"] not in used_recipes]
                    random.shuffle(available_meals)  # สุ่มลำดับเมนูเพื่อความหลากหลาย

                    for meal in available_meals:
                        if self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit):
                            daily_meals[i] = meal
                            used_recipes.add(meal["recipe_id"])  # บันทึกเมนูที่ใช้ไปแล้ว
                            break
            
            # คำนวณสารอาหารรวมของวันนั้น ๆ
            total_nutrition_per_day.append(self.calculate_total_nutrition(daily_meals))

        # เพิ่ม total_nutrition_per_day ลงใน JSON
        # mealplan["total_nutrition_per_day"] = total_nutrition_per_day
        return mealplan

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is running!"}
    
@app.get("/ai")
async def create_meals():
    print("🍽 กำลังดึงข้อมูลจากเซิร์ฟเวอร์...")

    creator = CreateMealPlan()
    food_data = creator.get_data()
    if not food_data:
        raise HTTPException(status_code=500, detail="Error fetching data")

    print("🔍 กำลังสร้างแผนมื้ออาหาร...")
    try:
        mealplan = creator.process_mealplan(food_data)
        creator.send_mealplan(mealplan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # return {"message": "แผนมื้ออาหารถูกสร้างเรียบร้อยแล้ว!"}
    return mealplan

@app.get("/ai_update")
async def update_meals():
    updater = UpdateMealPlan()
    print("🍽 กำลังดึงข้อมูลจากเซิร์ฟเวอร์...")

    try:
        food_data = updater.get_data()
        if not food_data:
            raise HTTPException(status_code=500, detail="ไม่สามารถดึงข้อมูลได้")
        
        print("🔍 กำลังสร้างแผนมื้ออาหาร...")
        mealplan = updater.get_mealplan()
        if not mealplan:
            raise HTTPException(status_code=500, detail="ไม่สามารถดึงแผนมื้ออาหารได้")
        
        updated_mealplan = updater.update_mealplan(mealplan, food_data["food_menus"], food_data["nutrition_limit_per_day"])
        updater.send_mealplan(updated_mealplan)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # return {"message": "แผนมื้ออาหารถูกอัปเดตเรียบร้อยแล้ว!"}
    return updated_mealplan


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 4000)))