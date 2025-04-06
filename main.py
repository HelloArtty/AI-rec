import json
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import logging
from typing import Dict, List, Set, Any, Tuple, Optional, Union
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()



# # ตั้งค่า logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

class NutritionCalculator:
    """
    คลาสสำหรับคำนวณและตรวจสอบข้อมูลทางโภชนาการ
    """
    
    @staticmethod
    def calculate_total_nutrition(meals: List[Dict]) -> Dict[str, float]:
        """
        คำนวณรวมค่าทางโภชนาการจากหลายมื้ออาหาร
        
        Args:
            meals: รายการของมื้ออาหาร แต่ละมื้อควรมี key "nutrition"
            
        Returns:
            ข้อมูลโภชนาการรวม
        """
        if not meals:
            return {}
            
        valid_meals = [meal for meal in meals if isinstance(meal, dict) and "nutrition" in meal]
        if not valid_meals:
            return {}

        total_nutrition = {key: 0 for key in valid_meals[0]["nutrition"].keys()}
        for meal in valid_meals:
            for key, value in meal.get("nutrition", {}).items():
                total_nutrition[key] += value
        return total_nutrition
    
    @staticmethod
    def is_within_nutrition_limit(
        meals: List[Dict], 
        nutrition_limit: Dict[str, float], 
        buffer_percentage: float = 0.08,
        min_calorie_percentage: float = 0.08
    ) -> bool:
        """
        ตรวจสอบว่าอาหารที่เลือกอยู่ในข้อจำกัดทางโภชนาการหรือไม่
        
        Args:
            meals: รายการของมื้ออาหาร
            nutrition_limit: ข้อจำกัดทางโภชนาการ
            buffer_percentage: เปอร์เซ็นต์เผื่อสำหรับค่าสูงสุด
            min_calorie_percentage: เปอร์เซ็นต์ขั้นต่ำของแคลอรี่ที่ต้องการ
            
        Returns:
            True หากอยู่ในข้อจำกัด, False หากไม่อยู่ในข้อจำกัด
        """
        total_nutrition = NutritionCalculator.calculate_total_nutrition(meals)
        
        # ตรวจสอบแคลอรี่ขั้นต่ำ
        min_calories = nutrition_limit.get("calories", 0) * min_calorie_percentage
        if total_nutrition.get("calories", 0) < min_calories:
            return False

        # ตรวจสอบข้อจำกัดโภชนาการแต่ละประเภท
        for key, limit in nutrition_limit.items():
            if limit == -1:  # ค่า -1 หมายถึงไม่มีข้อจำกัด
                continue
            buffer_limit = limit * (1 + buffer_percentage)
            if total_nutrition.get(key, 0) > buffer_limit:
                return False

        return True


class MealSelector:
    """
    คลาสสำหรับการเลือกอาหารที่เหมาะสม
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        กำหนดค่าเริ่มต้นสำหรับ MealSelector
        
        Args:
            random_seed: ค่า seed สำหรับการสุ่ม เพื่อให้ผลลัพธ์เหมือนกันในการทดสอบ
        """
        self.used_meals = defaultdict(list)
        if random_seed is not None:
            random.seed(random_seed)
    
    def weighted_random_selection(
        self, 
        meals: List[Dict], 
        key: str = "nutrition", 
        weight_key: str = "calories"
    ) -> Dict:
        """
        เลือกอาหารแบบสุ่มโดยใช้น้ำหนัก
        
        Args:
            meals: รายการอาหารที่จะเลือก
            key: คีย์หลักที่เก็บข้อมูลน้ำหนัก
            weight_key: คีย์ย่อยที่เก็บค่าน้ำหนัก
            
        Returns:
            อาหารที่ถูกเลือก
        """
        if not meals:
            raise ValueError("ไม่มีอาหารให้เลือก")
            
        weights = [meal.get(key, {}).get(weight_key, 0) for meal in meals]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(meals)  # fallback หากไม่มีน้ำหนัก
            
        probabilities = [weight / total_weight for weight in weights]
        return random.choices(meals, probabilities, k=1)[0]
    
    def can_use_meal(
        self,
        meal_name: str,
        current_day: int,
        max_reuse: int = 3,
        min_gap: int = 3
    ) -> bool:
        """
        ตรวจสอบว่าสามารถใช้อาหารในวันนี้ได้หรือไม่
        
        Args:
            meal_name: ชื่ออาหาร
            current_day: วันปัจจุบัน
            max_reuse: จำนวนครั้งสูงสุดที่อนุญาตให้ใช้อาหารนี้
            min_gap: จำนวนวันขั้นต่ำระหว่างการใช้อาหารชนิดเดียวกัน
            
        Returns:
            True หากสามารถใช้ได้, False หากไม่สามารถใช้ได้
        """
        if meal_name not in self.used_meals:
            return True
            
        usage_days = self.used_meals[meal_name]
        
        # ตรวจสอบจำนวนครั้งที่ใช้
        if len(usage_days) >= max_reuse:
            return False
            
        # ตรวจสอบระยะห่างระหว่างการใช้
        if any(abs(current_day - day) < min_gap for day in usage_days):
            return False
            
        return True
    
    def record_meal_usage(self, meal_name: str, day: int) -> None:
        """
        บันทึกการใช้อาหาร
        
        Args:
            meal_name: ชื่ออาหาร
            day: วันที่ใช้
        """
        self.used_meals[meal_name].append(day)
    
    def reset_meal_usage(self) -> None:
        """ล้างประวัติการใช้อาหารทั้งหมด"""
        self.used_meals.clear()


class MealPlanGenerator:
    """
    คลาสหลักสำหรับการสร้างแผนอาหาร
    """
    
    def __init__(
        self, 
        min_cal_percent: float = 0.20, 
        max_cal_ratio: float = 1.05,
        buffer_percentage: float = 0.08,
        random_seed: Optional[int] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ MealPlanGenerator
        
        Args:
            min_cal_percent: เปอร์เซ็นต์ขั้นต่ำของแคลอรี่ที่ต้องการ (0.0-1.0)
            max_cal_ratio: อัตราส่วนสูงสุดของแคลอรี่เทียบกับเป้าหมาย (เช่น 1.05 = 105%)
            buffer_percentage: เปอร์เซ็นต์เผื่อสำหรับค่าสูงสุด
            random_seed: ค่า seed สำหรับการสุ่ม
        """
        self.min_cal_percent = min_cal_percent
        self.max_cal_ratio = max_cal_ratio
        self.buffer_percentage = buffer_percentage
        self.nutrition_calculator = NutritionCalculator()
        self.meal_selector = MealSelector(random_seed)
    
    def filter_suitable_meals(
        self, 
        meals: List[Dict], 
        min_calories: float = 300
    ) -> List[Dict]:
        """
        กรองอาหารที่มีแคลอรี่ขั้นต่ำตามที่กำหนด
        
        Args:
            meals: รายการอาหารทั้งหมด
            min_calories: แคลอรี่ขั้นต่ำที่ต้องการ
            
        Returns:
            รายการอาหารที่ผ่านเกณฑ์
        """
        return [
            meal for meal in meals 
            if meal.get("nutrition", {}).get("calories", 0) >= min_calories
        ]
    
    def create_daily_meals(
        self, 
        all_meals: List[Dict], 
        nutrition_limit: Dict[str, float], 
        current_day: int,
        target_meal_count: int = 3,
        max_attempts: int = 500
    ) -> List[Dict]:
        """
        สร้างรายการอาหารสำหรับวันเดียว
        
        Args:
            all_meals: รายการอาหารทั้งหมด
            nutrition_limit: ข้อจำกัดทางโภชนาการ
            current_day: วันปัจจุบัน
            target_meal_count: จำนวนมื้ออาหารที่ต้องการ
            max_attempts: จำนวนครั้งสูงสุดที่จะพยายาม
            
        Returns:
            รายการอาหารสำหรับวันนี้
        """
        # กรองอาหารที่มีแคลอรี่เพียงพอ
        suitable_meals = self.filter_suitable_meals(all_meals)
        if not suitable_meals:
            # logger.warning("ไม่พบอาหารที่มีแคลอรี่เพียงพอ")
            return []
        
        
        daily_meals = []
        selected_meal_names = set()
        attempts = 0
        
        # คัดลอกรายการอาหารและสลับลำดับ
        available_meals = suitable_meals.copy()
        random.shuffle(available_meals)
        
        # พยายามเลือกอาหารจนกว่าจะได้จำนวนที่ต้องการ
        while len(daily_meals) < target_meal_count and attempts < max_attempts:
            attempts += 1
            
            if not available_meals:
                break

            # เลือกอาหารแบบถ่วงน้ำหนัก
            meal = self.meal_selector.weighted_random_selection(available_meals)
            
            # ตรวจสอบว่าอาหารนี้สามารถใช้ได้หรือไม่
            if meal["name"] not in selected_meal_names and self.meal_selector.can_use_meal(meal["name"], current_day):
                test_meals = daily_meals + [meal]
                if self.nutrition_calculator.is_within_nutrition_limit(test_meals, nutrition_limit, self.buffer_percentage):
                    meal_copy = meal.copy()
                    meal_copy["recipe_id"] = int(meal_copy["recipe_id"])
                    daily_meals.append(meal_copy)
                    selected_meal_names.add(meal["name"])
                    self.meal_selector.record_meal_usage(meal["name"], current_day)
            
            # ลบอาหารนี้ออกจากรายการที่พิจารณา
            available_meals.remove(meal)
        
        # ตัดอาหารส่วนเกินออกหากมีมากกว่า target_meal_count
        if len(daily_meals) > target_meal_count:
            daily_meals = daily_meals[:target_meal_count]
        
        return daily_meals
    def adjust_daily_meals(
        self, 
        daily_meals: List[Dict], 
        all_meals: List[Dict],
        nutrition_limit: Dict[str, float], 
        current_day: int,
        max_adjustment_attempts: int = 20
    ) -> List[Dict]:
        """
        ปรับแผนอาหารให้อยู่ในข้อจำกัดแคลอรี่
        
        Args:
            daily_meals: รายการอาหารปัจจุบัน
            all_meals: รายการอาหารทั้งหมด
            nutrition_limit: ข้อจำกัดทางโภชนาการ
            current_day: วันปัจจุบัน
            max_adjustment_attempts: จำนวนครั้งสูงสุดที่จะพยายามปรับแผน
            
        Returns:
            รายการอาหารที่ปรับแล้ว
        """
        calories_limit = nutrition_limit.get("calories", 0)
        min_calories = calories_limit * self.min_cal_percent
        max_calories = calories_limit * self.max_cal_ratio
        
        total_calories = sum(meal.get("nutrition", {}).get("calories", 0) for meal in daily_meals)
        
        attempts = 0
        while (total_calories < min_calories or total_calories > max_calories) and attempts < max_adjustment_attempts:
            if attempts == max_adjustment_attempts - 1:
                # logger.warning("Max adjustment attempts reached. Returning a meal plan that may not meet nutritional bounds.")
                break
            attempts += 1
            
            # หากแคลอรี่น้อยเกินไป ให้เพิ่มอาหาร
            if total_calories < min_calories:
                suitable_meals = self.filter_suitable_meals(all_meals)
                suitable_meals = [
                    meal for meal in suitable_meals 
                    if meal["name"] not in {m["name"] for m in daily_meals} and
                    self.meal_selector.can_use_meal(meal["name"], current_day)
                ]
                
                if not suitable_meals:
                    # logger.warning("ไม่มีอาหารที่เหมาะสมเหลือสำหรับเพิ่มแคลอรี่")
                    break
                    
                meal = self.meal_selector.weighted_random_selection(suitable_meals)
                meal_copy = meal.copy()
                meal_copy["recipe_id"] = int(meal_copy["recipe_id"])
                daily_meals.append(meal_copy)
                self.meal_selector.record_meal_usage(meal["name"], current_day)
                
            # หากแคลอรี่มากเกินไป ให้ลบอาหารที่มีแคลอรี่สูงออก
            elif total_calories > max_calories:
                sorted_meals = sorted(
                    daily_meals, 
                    key=lambda x: x.get("nutrition", {}).get("calories", 0),
                    reverse=True
                )
                removed_meal = sorted_meals[0]
                daily_meals.remove(removed_meal)
            
            # คำนวณแคลอรี่ใหม่
            total_calories = sum(meal.get("nutrition", {}).get("calories", 0) for meal in daily_meals)
        
        # ตัดอาหารส่วนเกินออกหากมีมากกว่า 3 มื้อ
        if len(daily_meals) > 3:
            daily_meals = daily_meals[:3]
        
        return daily_meals

    def create_meal_plan(
        self, 
        food_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        สร้างแผนอาหารตามข้อมูลที่กำหนด
        
        Args:
            food_data: ข้อมูลอาหารและการตั้งค่า
            
        Returns:
            แผนอาหารที่สร้างขึ้น
        """
        # ตรวจสอบรูปแบบข้อมูล
        if not isinstance(food_data, dict) or "food_menus" not in food_data:
            raise ValueError("รูปแบบข้อมูลไม่ถูกต้อง ต้องมี 'food_menus'")
            
        # ดึงข้อมูลจาก input
        food_menus = food_data["food_menus"]
        user_line_id = food_data.get("user_line_id", "")
        days = food_data.get("days")
        nutrition_limit = food_data.get("nutrition_limit_per_day", {})
        
        # ล้างประวัติการใช้อาหาร
        self.meal_selector.reset_meal_usage()
        
        # เตรียมผลลัพธ์
        mealplan = {
            "user_line_id": user_line_id,
            "mealplans": []
        }
        
        # สร้างแผนอาหารสำหรับแต่ละวัน
        for current_day in range(days):
            # logger.info(f"กำลังสร้างแผนอาหารสำหรับวันที่ {current_day + 1}")
            
            # สร้างอาหารรายวัน
            daily_meals = self.create_daily_meals(
                food_menus,
                nutrition_limit,
                current_day
            )
            
            # ตรวจสอบว่าได้อาหารเพียงพอหรือไม่
            if len(daily_meals) < 3:
                # logger.warning(f"วันที่ {current_day + 1} ไม่สามารถเลือกอาหารได้ครบ 3 มื้อ")
                pass
            
            # ปรับแผนอาหารให้อยู่ในข้อจำกัดแคลอรี่
            daily_meals = self.adjust_daily_meals(
                daily_meals,
                food_menus,
                nutrition_limit,
                current_day
            )
            
            # เพิ่มแผนอาหารลงในผลลัพธ์
            mealplan["mealplans"].append(daily_meals)
            
            # แสดงข้อมูลแคลอรี่รายวัน
            daily_calories = sum(meal.get("nutrition", {}).get("calories", 0) for meal in daily_meals)
            # logger.info(f"วันที่ {current_day + 1} - แคลอรี่รวม: {daily_calories}")
        
        # คำนวณโภชนาการรวม
        total_nutrition = self.nutrition_calculator.calculate_total_nutrition(
            [meal for daily in mealplan["mealplans"] for meal in daily]
        )
        total_nutrition = {key: round(value, 2) for key, value in total_nutrition.items()}
        
        # แสดงข้อมูลสรุป
        total_calories = total_nutrition.get("calories", 0)
        max_cals = nutrition_limit.get("calories", 0) * self.max_cal_ratio
        min_cals = nutrition_limit.get("calories", 0) * self.min_cal_percent
        
        # logger.info(f"แคลอรี่รวม (ทุกวัน): {total_calories}")
        # logger.info(f"แคลอรี่ขั้นต่ำที่อนุญาต: {round(min_cals, 2)}")
        # logger.info(f"แคลอรี่สูงสุดที่อนุญาต: {round(max_cals, 2)}")
        
        # ตรวจสอบและแสดงข้อมูลอาหารที่ใช้บ่อยที่สุด
        # meal_usage = {meal: len(days) for meal, days in self.meal_selector.used_meals.items()}
        # most_used_meals = sorted(meal_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # logger.info("อาหารที่ใช้บ่อยที่สุด:")
        # for meal_name, count in most_used_meals:
            # logger.info(f"- {meal_name}: {count} ครั้ง")
        # logger.info("✅ สร้างแผนอาหารสำเร็จ!")

        print("📊 Total Calories:", round(total_calories, 2))
        print("📊 Max Calories:", round(max_cals, 2))
        print("📊 Min Calories:", round(min_cals, 2))
        
        print("📌 Most Used Meals:")
        meal_usage = {meal: len(days) for meal, days in self.meal_selector.used_meals.items()}
        most_used_meals = sorted(meal_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        for meal_name, count in most_used_meals:
            print(f"- {meal_name}: {count} ครั้ง")
            
            
        print("✅ Successfully created meal plan!")
        
        return mealplan


class CreateMealPlan(MealPlanGenerator):
    """
    คลาสที่รักษาชื่อเดิมเพื่อความเข้ากันได้กับโค้ดเดิม
    """
    
    def __init__(self):
        """ใช้ค่าเริ่มต้นเหมือนโค้ดเดิม"""
        super().__init__(
            min_cal_percent=0.85,
            max_cal_ratio=1.05,
            buffer_percentage=0.08
        )
        self.cal_limit = 0.85  # สำหรับความเข้ากันได้กับโค้ดเดิม
        self.per = 0.20        # สำหรับความเข้ากันได้กับโค้ดเดิม
        self.max = 1.05        # สำหรับความเข้ากันได้กับโค้ดเดิม
    
    def process_mealplan(self, food_data):
        """
        สร้างแผนอาหารโดยใช้ method ใหม่
        """
        return self.create_meal_plan(food_data)
        
    # ใส่ method จากโค้ดเดิมเพื่อความเข้ากันได้
    @staticmethod
    def calculate_total_nutrition(meals):
        return NutritionCalculator.calculate_total_nutrition(meals)
        
    @staticmethod
    def is_within_nutrition_limit(meals, nutrition_limit, buffer_percentage=0.08):
        return NutritionCalculator.is_within_nutrition_limit(
            meals, nutrition_limit, buffer_percentage
        )




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