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

class CreateMealPlan:
    def __init__(self):
        self.used_meals = set()

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
    def is_within_nutrition_limit(meals, nutrition_limit, buffer_percentage=0.08):
        """ตรวจสอบว่าอาหารยังอยู่ในขอบเขตโภชนาการ (ยืดหยุ่นขึ้น)"""
        total_nutrition = CreateMealPlan.calculate_total_nutrition(meals)

        # เช็คว่า calories รวมของ 3 มื้อไม่ต่ำกว่า 8% ของ limit calories
        min_calories = nutrition_limit.get("calories", 0) * 0.08  # คำนวณ 8% ของ limit calories
        if total_nutrition.get("calories", 0) < min_calories:
            # print(f"❌ Total calories {total_nutrition.get('calories', 0)} is below 8% of the limit.")
            return False  # ถ้าค่า calories รวมต่ำกว่า 8% ของ limit calories ให้ไม่ผ่าน

        # เช็คขอบเขตโภชนาการอื่น ๆ
        for key, limit in nutrition_limit.items():
            if limit == -1:
                continue  # ถ้าค่าจำกัดเป็น -1 หมายถึงไม่จำกัด ให้ข้ามไป
            buffer_limit = limit * (1 + buffer_percentage)  # เพิ่ม buffer 8%
            if total_nutrition.get(key, 0) > buffer_limit:
                return False  # ถ้าเกินขอบเขตที่ยืดหยุ่นแล้ว ไม่ให้ผ่าน

        return True  # ถ้าผ่านทุกตัวแปรโภชนาการ แสดงว่าผ่าน

    def select_meal(self, cluster_meals, selected_meals):
        """เลือกเมนูจากคลัสเตอร์ที่ยังไม่ซ้ำ"""
        random.shuffle(cluster_meals)
        for meal in cluster_meals:
            if meal["name"] not in selected_meals and meal["name"] not in self.used_meals:
                # เช็คว่า meal มีแคลอรีมากกว่า 100 หรือไม่
                if meal["nutrition"].get("calories", 0) > 300:
                    return meal
        return None

    def cluster_meals(self, food_menus):
        """แบ่งกลุ่มอาหารโดยใช้ KMeans"""
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

        return clustered_meals

    def adjust_meals_based_on_calories(self, daily_meals, nutrition_limit, clustered_meals):
        """ปรับมื้ออาหารให้ตรงตามเงื่อนไขของแคลอรี"""
        total_calories = sum(meal["nutrition"].get("calories", 0) for meal in daily_meals)
        selected_meals = set()
        calories_limit = nutrition_limit.get("calories", 0)
        min_8 = calories_limit * 0.15
        min_calories = max(calories_limit - min_8, 0)
        max_calories = calories_limit * 1.02

        attempts = 0  # ตัวนับจำนวนรอบการปรับ

        # หากแคลอรีไม่ตรงตามเงื่อนไข
        while total_calories < min_calories or total_calories > max_calories:
            attempts += 1  # เพิ่มตัวนับรอบ
            # print(f"🔄 Adjusting meals... Attempt #{attempts}")

            daily_meals.clear()
            selected_meals.clear()

            # เลือกมื้ออาหารจากคลัสเตอร์ใหม่
            random.shuffle(list(clustered_meals.values()))
            for cluster_id in clustered_meals:
                meal = self.select_meal(clustered_meals[cluster_id], selected_meals)
                if meal:
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            remaining_meals = [meal for cluster in clustered_meals.values() for meal in cluster]
            random.shuffle(remaining_meals)

            while len(daily_meals) < 3 and remaining_meals:
                meal = remaining_meals.pop(0)
                if meal["name"] not in selected_meals and self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit):
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            # ตรวจสอบแคลอรีใหม่
            total_calories = sum(meal["nutrition"].get("calories", 0) for meal in daily_meals)

        print(f"✅ MealPlan (PD) successfully in {attempts} rounds.")
        return daily_meals

    def process_mealplan(self, food_data):
        """สร้างแผนมื้ออาหารแบบไม่ให้ซ้ำกันมากเกินไป"""
        if not isinstance(food_data, dict) or "food_menus" not in food_data:
            raise ValueError("Invalid food_data format")

        food_menus = food_data["food_menus"]
        user_line_id = food_data.get("user_line_id", "")
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
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            remaining_meals = [meal for cluster in clustered_meals.values() for meal in cluster]
            random.shuffle(remaining_meals)

            # หามื้ออาหารจนกว่าจะตรงตามเงื่อนไข
            while len(daily_meals) < 3 and remaining_meals:
                meal = remaining_meals.pop(0)
                if meal["name"] not in selected_meals and self.is_within_nutrition_limit(daily_meals + [meal], nutrition_limit):
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            # ตรวจสอบ total calories กับ nutrition limit
            daily_meals = self.adjust_meals_based_on_calories(daily_meals, nutrition_limit, clustered_meals)

            mealplan["mealplans"].append(daily_meals)


        # Print total calories per day
        for day, daily_meals in enumerate(mealplan["mealplans"], start=1):
            daily_calories = sum(meal["nutrition"].get("calories", 0) for meal in daily_meals)
            print(f"📅 Day {day} - Total Calories: {daily_calories}")

        # Calculate total nutrition for all days
        total_nutrition = self.calculate_total_nutrition([meal for daily_meals in mealplan["mealplans"] for meal in daily_meals])
        total_nutrition = {key: round(value, 2) for key, value in total_nutrition.items()}  # Round to 2 decimal places

        # Extract calorie-related information
        total_calories = total_nutrition.get("calories", 0)
        max_calories = nutrition_limit.get("calories", 0) * 1.02  # 102% of the limit
        min_calories = max(nutrition_limit.get("calories", 0) * 0.85, 0)  # 85% of the limit

        # Print overall calorie information
        print(f"📊 Total Calories (All Days): {total_calories}")
        print(f"📊 Max Calories Allowed: {round(max_calories, 2)}")
        print(f"📊 Min Calories Allowed: {round(min_calories, 2)}")

        return mealplan

class UpdateMealPlan:
    def __init__(self):
        super().__init__()
        # Initialize used_meals set to track used meal names across days
        self.used_meals = set()

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
    def is_within_nutrition_limit(meals, nutrition_limit, buffer_percentage=0.08):
        """ตรวจสอบว่าโภชนาการอยู่ในขอบเขตที่กำหนด พร้อม buffer"""
        total_nutrition = UpdateMealPlan.calculate_total_nutrition(meals)

        # เช็คว่า calories รวมของมื้อไม่ต่ำกว่า 85% ของ limit calories
        min_calories = nutrition_limit.get("calories", 0) * 0.85  # คำนวณ 85% ของ limit calories
        if total_nutrition.get("calories", 0) < min_calories:
            return False  # ถ้าค่า calories รวมต่ำกว่า 85% ของ limit calories ให้ไม่ผ่าน

        # เช็คว่า calories รวมไม่เกิน 102% ของ limit calories
        max_calories = nutrition_limit.get("calories", 0) * 1.02  # คำนวณ 102% ของ limit calories
        if total_nutrition.get("calories", 0) > max_calories:
            return False  # ถ้าค่า calories รวมเกิน 102% ของ limit calories ให้ไม่ผ่าน

        # เช็คขอบเขตโภชนาการอื่น ๆ
        for key, limit in nutrition_limit.items():
            if key == "calories":
                continue  # เราได้ตรวจสอบแคลอรี่แล้ว
            if limit == -1:
                continue  # ถ้าค่าจำกัดเป็น -1 หมายถึงไม่จำกัด ให้ข้ามไป
            buffer_limit = limit * (1 + buffer_percentage)  # เพิ่ม buffer 8%
            if total_nutrition.get(key, 0) > buffer_limit:
                return False  # ถ้าเกินขอบเขตที่ยืดหยุ่นแล้ว ไม่ให้ผ่าน

        return True  # ถ้าผ่านทุกตัวแปรโภชนาการ แสดงว่าผ่าน

    def select_meal(self, cluster_meals, selected_meals):
        """เลือกเมนูจากคลัสเตอร์ที่ยังไม่ซ้ำ และมีแคลอรี่เพียงพอ"""
        random.shuffle(cluster_meals)
        for meal in cluster_meals:
            # ตรวจสอบว่าเมนูยังไม่ถูกใช้ในมื้อปัจจุบันและวันอื่น ๆ
            if meal["name"] not in selected_meals and meal["name"] not in self.used_meals:
                # เช็คว่า meal มีแคลอรีมากกว่า 300 หรือไม่
                if meal["nutrition"].get("calories", 0) > 300:
                    return meal
        
        # ถ้าไม่พบเมนูที่มีแคลอรี่มากกว่า 300 ให้พยายามหาเมนูที่ยังไม่ได้ใช้ (ไม่สนใจแคลอรี่)
        for meal in cluster_meals:
            if meal["name"] not in selected_meals and meal["name"] not in self.used_meals:
                return meal
                
        # ถ้ายังไม่พบเมนูที่ยังไม่ถูกใช้ ให้ใช้เมนูซ้ำที่ยังไม่ได้ใช้ในมื้อปัจจุบัน
        for meal in cluster_meals:
            if meal["name"] not in selected_meals:
                return meal
                
        return None

    def cluster_meals(self, food_menus):
        """แบ่งกลุ่มอาหารโดยใช้ KMeans"""
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

    def adjust_meals_based_on_calories(self, daily_meals, nutrition_limit, clustered_meals):
        """ปรับมื้ออาหารให้ตรงตามเงื่อนไขของแคลอรี"""
        selected_meals = set()
        calories_limit = nutrition_limit.get("calories", 0)
        min_calories = max(calories_limit * 0.85, 0)  # 85% ของ limit
        max_calories = calories_limit * 1.02  # 102% ของ limit

        attempts = 0  # ตัวนับจำนวนรอบการปรับ
        max_attempts = 2000  # กำหนดจำนวนรอบสูงสุด เพื่อป้องกัน infinite loop

        # เตรียมรายการเมนูที่มีแคลอรี่สูง
        high_calorie_meals = []
        for cluster in clustered_meals.values():
            for meal in cluster:
                if meal["nutrition"].get("calories", 0) > 300:  # เลือกเมนูที่มีแคลอรี่สูง
                    high_calorie_meals.append(meal)

        while attempts < max_attempts:
            attempts += 1
            daily_meals.clear()
            selected_meals.clear()

            # เลือกมื้ออาหารจากคลัสเตอร์ใหม่
            cluster_list = list(clustered_meals.keys())
            random.shuffle(cluster_list)
            
            # ถ้าแคลอรี่ต่ำ พยายามเลือกเมนูที่มีแคลอรี่สูง 1 เมนู
            if attempts > 10 and high_calorie_meals:
                meal = random.choice(high_calorie_meals)
                meal["recipe_id"] = int(meal["recipe_id"])
                daily_meals.append(meal)
                selected_meals.add(meal["name"])
                
                # ลดจำนวนคลัสเตอร์ลง 1 เพราะใช้เมนูแคลอรี่สูงแล้ว 1 เมนู
                if len(cluster_list) > 0:
                    cluster_list.pop()
            
            # เลือกเมนูจากแต่ละคลัสเตอร์ที่เหลือ
            for cluster_id in cluster_list:
                meal = self.select_meal(clustered_meals[cluster_id], selected_meals)
                if meal:
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            # ถ้ายังไม่ครบ 3 มื้อ เลือกเพิ่มจากเมนูที่เหลือ
            remaining_meals = [meal for cluster in clustered_meals.values() for meal in cluster]
            random.shuffle(remaining_meals)

            while len(daily_meals) < 3 and remaining_meals:
                meal = remaining_meals.pop(0)
                if meal["name"] not in selected_meals:
                    meal["recipe_id"] = int(meal["recipe_id"])
                    daily_meals.append(meal)
                    selected_meals.add(meal["name"])
                    self.used_meals.add(meal["name"])

            # ตรวจสอบแคลอรีใหม่
            total_calories = sum(meal["nutrition"].get("calories", 0) for meal in daily_meals)
            
            # เช็คว่าอยู่ในช่วงที่ต้องการหรือไม่
            if min_calories <= total_calories <= max_calories:
                print(f"✅ MealPlan successfully adjusted in {attempts} rounds. Total calories: {total_calories}")
                return daily_meals

        # ถ้าไม่สามารถหาเมนูที่เหมาะสมได้หลังจากลองหลายรอบ ให้เลือกเมนูที่ใกล้เคียงที่สุด
        print(f"⚠️ Could not find ideal meal plan after {max_attempts} attempts. Using best available.")
        return daily_meals

    def update_mealplan(self, mealplan, food_menus, nutrition_limit):
        """อัปเดตแผนมื้ออาหารโดยเลือกเมนูที่สมดุลที่สุด"""
        # ล้าง used_meals เมื่อเริ่มต้นการอัปเดตใหม่
        self.used_meals = set()
        
        clustered_meals = self.cluster_meals(food_menus)

        for day_index, daily_meals in enumerate(mealplan["mealplans"]):
            # Ensure daily_meals is a list
            if not isinstance(daily_meals, list):
                print(f"❌ Invalid daily_meals format: {daily_meals}")
                continue  # Skip invalid entries

            used_recipes = {int(meal["recipe_id"]) for meal in daily_meals if isinstance(meal, dict) and "recipe_id" in meal}
            empty_slots = [i for i, meal in enumerate(daily_meals) if meal == {}]

            available_meals = [meal for meal in food_menus if int(meal["recipe_id"]) not in used_recipes]
            
            # เก็บเมนูเดิมที่มีอยู่แล้ว (ไม่ใช่ empty slot)
            existing_meals = [meal for meal in daily_meals if meal != {}]
            
            # เริ่มด้วยการตรวจสอบว่าเมนูที่มีอยู่แล้วอยู่ในขอบเขตหรือไม่
            if existing_meals and not self.is_within_nutrition_limit(existing_meals, nutrition_limit):
                # ถ้าไม่อยู่ในขอบเขต ให้เริ่มการเลือกเมนูใหม่ทั้งหมด
                daily_meals = []
                new_meals = self.adjust_meals_based_on_calories(daily_meals, nutrition_limit, clustered_meals)
                # อัปเดตเมนูในวันนั้น
                mealplan["mealplans"][day_index] = new_meals
            else:
                # ถ้ามี empty slots ให้เติมเมนูใหม่ที่ไม่ทำให้เกินขอบเขต
                for i in empty_slots:
                    found_suitable_meal = False
                    random.shuffle(available_meals)
                    
                    for meal in available_meals:
                        temp_meals = existing_meals.copy()
                        temp_meals.append(meal)
                        if self.is_within_nutrition_limit(temp_meals, nutrition_limit):
                            meal["recipe_id"] = int(meal["recipe_id"])
                            daily_meals[i] = meal
                            self.used_meals.add(meal["name"])
                            used_recipes.add(meal["recipe_id"])
                            available_meals.remove(meal)
                            found_suitable_meal = True
                            break
                    
                    # ถ้าไม่พบเมนูที่เหมาะสม ให้ลองเลือกจากคลัสเตอร์
                    if not found_suitable_meal:
                        for cluster_id in clustered_meals:
                            cluster_meals = clustered_meals[cluster_id]
                            random.shuffle(cluster_meals)
                            for meal in cluster_meals:
                                if int(meal["recipe_id"]) not in used_recipes:
                                    temp_meals = existing_meals.copy()
                                    temp_meals.append(meal)
                                    if self.is_within_nutrition_limit(temp_meals, nutrition_limit):
                                        meal["recipe_id"] = int(meal["recipe_id"])
                                        daily_meals[i] = meal
                                        self.used_meals.add(meal["name"])
                                        used_recipes.add(meal["recipe_id"])
                                        found_suitable_meal = True
                                        break
                            if found_suitable_meal:
                                break
                
                # หลังจากเติมเมนูแล้ว ตรวจสอบอีกครั้งว่าอยู่ในขอบเขตหรือไม่
                if not self.is_within_nutrition_limit(daily_meals, nutrition_limit):
                    print(f"❌ Day {day_index+1}: Nutrition limit not satisfied. Adjusting meals completely.")
                    new_meals = self.adjust_meals_based_on_calories([], nutrition_limit, clustered_meals)
                    mealplan["mealplans"][day_index] = new_meals

            # Print total calories per day
            # daily_calories = sum(meal["nutrition"].get("calories", 0) for meal in mealplan["mealplans"][day_index])
            # print(f"📅 Day {day_index+1} - Total Calories: {daily_calories}")

        # Calculate total nutrition for all days
        total_nutrition = self.calculate_total_nutrition([meal for daily_meals in mealplan["mealplans"] for meal in daily_meals])
        total_nutrition = {key: round(value, 2) for key, value in total_nutrition.items()}  # Round to 2 decimal places

        # Extract calorie-related information
        total_calories = total_nutrition.get("calories", 0)
        max_calories = nutrition_limit.get("calories", 0) * 1.02  # 102% of the limit
        min_calories = max(nutrition_limit.get("calories", 0) * 0.85, 0)  # 85% of the limit

        # Print overall calorie information
        print(f"📊 Total Calories (All Days): {total_calories}")
        print(f"📊 Max Calories Allowed: {round(max_calories, 2)}")
        print(f"📊 Min Calories Allowed: {round(min_calories, 2)}")

        return mealplan


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
    uvicorn.run("main:app", host="127.0.0.1", port=4000, reload=True)