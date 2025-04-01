import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

class MealPlanner(ABC):
    def __init__(self):
        super().__init__()

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
    def is_within_nutrition_limit(meals, nutrition_limit, buffer_percentage=0.3):
        """ ตรวจสอบว่าอาหารยังอยู่ในขอบเขตโภชนาการ (ยืดหยุ่นขึ้น) """
        total_nutrition = MealPlanner.calculate_total_nutrition(meals)
        for key, limit in nutrition_limit.items():
            if limit == -1:
                continue  # ถ้าค่าจำกัดเป็น -1 หมายถึงไม่จำกัด ให้ข้ามไป
            buffer_limit = limit * (1 + buffer_percentage)  # เพิ่ม buffer 30%
            if total_nutrition.get(key, 0) > buffer_limit:
                return False  # ถ้าเกินขอบเขตที่ยืดหยุ่นแล้ว ไม่ให้ผ่าน
        return True  # ถ้าผ่านทุกตัวแปรโภชนาการ แสดงว่าผ่าน

    @abstractmethod
    def process_mealplan(self, *args, **kwargs):
        """Abstract method to be implemented by child classes"""
        pass


class CreateMealPlan(MealPlanner):
    def __init__(self):
        super().__init__()
        self.used_meals = set()
        
    def select_meal(self, cluster_meals, selected_meals):
        """เลือกเมนูจากคลัสเตอร์ที่ยังไม่ซ้ำ"""
        random.shuffle(cluster_meals)
        for meal in cluster_meals:
            if meal["name"] not in selected_meals and meal["name"] not in self.used_meals:
                return meal
        return None
    
    def cluster_meals(self, food_menus):
        """ แบ่งกลุ่มอาหารโดยใช้ KMeans """
        num_clusters = 3  # บังคับให้แบ่งเป็น 3 คลัสเตอร์เสมอ
        model = SentenceTransformer('all-MiniLM-L6-v2')

        meal_names = [meal['name'] for meal in food_menus]
        embeddings = model.encode(meal_names)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        clustered_meals = {i: [] for i in range(num_clusters)}
        for meal, cluster in zip(food_menus, clusters):
            clustered_meals[cluster].append(meal)

        # ตรวจสอบจำนวนเมนูทั้งหมดแล้วหาร 3 เพื่อกำหนดจำนวนเมนูในแต่ละคลัสเตอร์
        num_meals = len(food_menus)
        print("📌 Number of meals:", num_meals)
        min_meals_per_cluster = num_meals // num_clusters
        print("📌 Min meals per cluster:", min_meals_per_cluster)

        # ✅ ถ้าคลัสเตอร์ไหนไม่มีเมนู → แบ่งใหม่แบบสุ่ม
        if any(len(v) < min_meals_per_cluster for v in clustered_meals.values()):
            # print("⚠️ Clustering is imbalanced. Using random splitting instead.")
            random.shuffle(food_menus)
            clustered_meals = {
                0: food_menus[:num_meals // 3],
                1: food_menus[num_meals // 3 : 2 * num_meals // 3],
                2: food_menus[2 * num_meals // 3:]
            }
        
        return clustered_meals

    def process_mealplan(self, food_data):
        """สร้างแผนมื้ออาหารแบบไม่ให้ซ้ำกันมากเกินไป"""
        if not isinstance(food_data, dict) or "food_menus" not in food_data:
            raise ValueError("Invalid food_data format")
        
        food_menus = food_data["food_menus"]
        user_line_id = int(food_data.get("user_line_id", ""))
        days = food_data.get("days")
        nutrition_limit = food_data.get("nutrition_limit_per_day", {})

        clustered_meals = self.cluster_meals(food_menus)
        print("📌 Clustered Meals:", {k: len(v) for k, v in clustered_meals.items()})
        mealplan = {"user_line_id": user_line_id, "mealplans": []}

        for _ in range(days):
            daily_meals = []
            selected_meals = set()

            for cluster_id in clustered_meals:
                meal = self.select_meal(clustered_meals[cluster_id], selected_meals)
                if meal:
                    meal["recipe_id"] = int(meal["recipe_id"])  # แปลง recipe_id เป็น int
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            remaining_meals = [meal for cluster in clustered_meals.values() for meal in cluster]
            random.shuffle(remaining_meals)
            while len(daily_meals) < 3 and remaining_meals:
                meal = remaining_meals.pop(0)
                if meal["name"] not in selected_meals and self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit):
                    meal["recipe_id"] = int(meal["recipe_id"])  # แปลง recipe_id เป็น int
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            mealplan["mealplans"].append(daily_meals)

        return mealplan


class UpdateMealPlan:
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_total_nutrition(meals):
        """คำนวณโภชนาการรวมของมื้ออาหาร"""
        valid_meals = [meal for meal in meals if isinstance(meal, dict) and "nutrition" in meal]
        if not valid_meals:
            return {}

        total_nutrition = {key: 0 for key in valid_meals[0]["nutrition"].keys()}
        for meal in valid_meals:
            for key, value in meal["nutrition"].items():
                total_nutrition[key] += value
        
        return total_nutrition

    @staticmethod
    def is_within_nutrition_limit(meals, nutrition_limit, buffer_percentage=0.3):
        """ตรวจสอบว่าโภชนาการอยู่ในขอบเขตที่กำหนด พร้อม buffer"""
        total_nutrition = UpdateMealPlan.calculate_total_nutrition(meals)
        for key, limit in nutrition_limit.items():
            if limit == -1:  
                continue  
            buffer_limit = limit * (1 + buffer_percentage)
            if total_nutrition.get(key, 0) > buffer_limit:
                return False
        return True

    def update_mealplan(self, mealplan, food_menus, nutrition_limit):
        """อัปเดตแผนมื้ออาหารโดยเลือกเมนูที่สมดุลที่สุด"""
        clustered_meals = self.cluster_meals(food_menus)

        for daily_meals in mealplan["mealplans"]:
            used_recipes = {int(meal["recipe_id"]) for meal in daily_meals if isinstance(meal, dict) and "recipe_id" in meal}
            empty_slots = [i for i, meal in enumerate(daily_meals) if meal == {}]

            available_meals = [meal for meal in food_menus if int(meal["recipe_id"]) not in used_recipes]
            random.shuffle(available_meals)

            for i in empty_slots:
                for meal in available_meals:
                    if self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit):
                        meal["recipe_id"] = int(meal["recipe_id"])
                        daily_meals[i] = meal
                        used_recipes.add(int(meal["recipe_id"]))
                        available_meals.remove(meal)
                        break

            for i in empty_slots:
                if daily_meals[i] == {}:
                    for cluster_id in clustered_meals:
                        cluster_meals = clustered_meals[cluster_id]
                        random.shuffle(cluster_meals)
                        for meal in cluster_meals:
                            if int(meal["recipe_id"]) not in used_recipes:
                                meal["recipe_id"] = int(meal["recipe_id"])
                                daily_meals[i] = meal
                                used_recipes.add(int(meal["recipe_id"]))
                                break
                        if daily_meals[i] != {}:
                            break

        return mealplan
    
    def cluster_meals(self, food_menus):
        """ แบ่งกลุ่มอาหารโดยใช้ KMeans """
        num_clusters = 3  # บังคับให้แบ่งเป็น 3 คลัสเตอร์เสมอ
        model = SentenceTransformer('all-MiniLM-L6-v2')

        meal_names = [meal['name'] for meal in food_menus]
        embeddings = model.encode(meal_names)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        clustered_meals = {i: [] for i in range(num_clusters)}
        for meal, cluster in zip(food_menus, clusters):
            clustered_meals[cluster].append(meal)

        # ตรวจสอบจำนวนเมนูทั้งหมดแล้วหาร 3 เพื่อกำหนดจำนวนเมนูในแต่ละคลัสเตอร์
        num_meals = len(food_menus)
        print("📌 Number of meals:", num_meals)
        min_meals_per_cluster = num_meals // num_clusters
        print("📌 Min meals per cluster:", min_meals_per_cluster)

        # ✅ ถ้าคลัสเตอร์ไหนไม่มีเมนู → แบ่งใหม่แบบสุ่ม
        if any(len(v) < min_meals_per_cluster for v in clustered_meals.values()):
            random.shuffle(food_menus)
            clustered_meals = {
                0: food_menus[:num_meals // 3],
                1: food_menus[num_meals // 3 : 2 * num_meals // 3],
                2: food_menus[2 * num_meals // 3:]
            }
        print("📌 Clustered Meals:", {k: len(v) for k, v in clustered_meals.items()})
        return clustered_meals

    def find_balanced_meal(self, available_meals, nutrition_limit):
        """ค้นหาเมนูที่สมดุลที่สุดจากรายการที่เหลือ"""
        best_meal = None
        best_score = float("inf")

        for meal in available_meals:
            total_nutrition = self.calculate_total_nutrition([meal])
            score = sum(abs((total_nutrition.get(nutr, 0) / nutrition_limit[nutr]) - 1) for nutr in nutrition_limit if nutrition_limit[nutr] > 0)
            
            if score < best_score:
                best_score = score
                best_meal = meal

        return best_meal



app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running!"}
    
@app.post("/ai")
async def create_meals(request: Request):
    print("🍽 กำลังดึงข้อมูลจากเซิร์ฟเวอร์...")

    creator = CreateMealPlan()
    food_data = await request.json()
    
    if not food_data:
        raise HTTPException(status_code=400, detail="Invalid input data")

    print("🔍 กำลังสร้างแผนมื้ออาหาร...")
    try:
        mealplan = creator.process_mealplan(food_data)
    except Exception as e:
        print(f"❌ Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return mealplan

@app.post("/ai_update")
async def update_meals(request: Request):
    updater = UpdateMealPlan()
    print("🍽 กำลังดึงข้อมูลจากเซิร์ฟเวอร์...")

    try:
        request_data = await request.json()
        if not request_data:
            raise HTTPException(status_code=400, detail="Invalid input data")
        
        food_data = request_data.get("food_menus")
        nutrition_limit_per_day = request_data.get("nutrition_limit_per_day")
        mealplan = request_data.get("mealplan")
        
        if not food_data or not nutrition_limit_per_day or not mealplan:
            raise HTTPException(status_code=400, detail="Missing required data")
        
        print("🔍 กำลังสร้างแผนมื้ออาหาร...")
        updated_mealplan = updater.update_mealplan(mealplan, food_data, nutrition_limit_per_day)
    
    except Exception as e:
        print(f"❌ Error updating meal plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return updated_mealplan

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)