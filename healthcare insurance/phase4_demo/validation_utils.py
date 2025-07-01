#!/usr/bin/env python3
"""
医疗信息验证工具
验证从表单中抽取的关键信息的格式和有效性
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import phonenumbers
from phonenumbers import geocoder, carrier
import datetime
import json

# 设置日志
logger = logging.getLogger(__name__)

class MedicalFieldValidator:
    """医疗字段验证器"""
    
    def __init__(self):
        # 美国区号列表（部分）
        self.us_area_codes = {
            '201', '202', '203', '205', '206', '207', '208', '209', '210',
            '212', '213', '214', '215', '216', '217', '218', '219', '224',
            '225', '228', '229', '231', '234', '239', '240', '248', '251',
            '252', '253', '254', '256', '260', '262', '267', '269', '270',
            '276', '281', '301', '302', '303', '304', '305', '307', '308',
            '309', '310', '312', '313', '314', '315', '316', '317', '318',
            '319', '320', '321', '323', '325', '330', '331', '334', '336',
            '337', '339', '347', '351', '352', '360', '361', '386', '401',
            '402', '404', '405', '406', '407', '408', '409', '410', '412',
            '413', '414', '415', '417', '419', '423', '424', '425', '430',
            '432', '434', '435', '440', '443', '445', '464', '469', '470',
            '475', '478', '479', '480', '484', '501', '502', '503', '504',
            '505', '507', '508', '509', '510', '512', '513', '515', '516',
            '517', '518', '520', '530', '540', '541', '551', '559', '561',
            '562', '563', '564', '567', '570', '571', '573', '574', '575',
            '580', '585', '586', '601', '602', '603', '605', '606', '607',
            '608', '609', '610', '612', '614', '615', '616', '617', '618',
            '619', '620', '623', '626', '630', '631', '636', '641', '646',
            '650', '651', '660', '661', '662', '667', '678', '682', '701',
            '702', '703', '704', '706', '707', '708', '712', '713', '714',
            '715', '716', '717', '718', '719', '720', '724', '727', '731',
            '732', '734', '737', '740', '747', '754', '757', '760', '763',
            '765', '770', '772', '773', '774', '775', '781', '785', '786',
            '801', '802', '803', '804', '805', '806', '808', '810', '812',
            '813', '814', '815', '816', '817', '818', '828', '830', '831',
            '832', '843', '845', '847', '848', '850', '856', '857', '858',
            '859', '860', '862', '863', '864', '865', '870', '872', '878',
            '901', '903', '904', '906', '907', '908', '909', '910', '912',
            '913', '914', '915', '916', '917', '918', '919', '920', '925',
            '928', '929', '931', '936', '937', '940', '941', '947', '949',
            '951', '952', '954', '956', '970', '971', '972', '973', '978',
            '979', '980', '985', '989'
        }
        
        # ICD-10诊断代码格式
        self.icd10_pattern = re.compile(r'^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$')
        
        # 常见的NPI号码格式
        self.npi_pattern = re.compile(r'^\d{10}$')
        
    def validate_phone_number(self, phone: str) -> Dict[str, any]:
        """验证电话号码"""
        result = {
            'is_valid': False,
            'formatted': None,
            'area_code': None,
            'location': None,
            'carrier': None,
            'errors': []
        }
        
        try:
            # 尝试解析电话号码
            parsed = phonenumbers.parse(phone, 'US')
            
            if phonenumbers.is_valid_number(parsed):
                result['is_valid'] = True
                result['formatted'] = phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.NATIONAL
                )
                
                # 获取区号
                area_code = str(parsed.national_number)[:3]
                result['area_code'] = area_code
                
                # 检查区号是否在美国区号列表中
                if area_code not in self.us_area_codes:
                    result['errors'].append(f'区号 {area_code} 不在美国有效区号列表中')
                
                # 获取地理位置
                try:
                    location = geocoder.description_for_number(parsed, 'en')
                    result['location'] = location
                except:
                    pass
                
                # 获取运营商信息
                try:
                    carrier_name = carrier.name_for_number(parsed, 'en')
                    result['carrier'] = carrier_name
                except:
                    pass
            else:
                result['errors'].append('电话号码格式无效')
                
        except phonenumbers.NumberParseException as e:
            result['errors'].append(f'电话号码解析失败: {e}')
        except Exception as e:
            result['errors'].append(f'电话号码验证异常: {e}')
        
        return result
    
    def validate_npi(self, npi: str) -> Dict[str, any]:
        """验证NPI号码"""
        result = {
            'is_valid': False,
            'formatted': None,
            'errors': []
        }
        
        # 清理NPI号码
        clean_npi = re.sub(r'[^\d]', '', npi)
        
        if not self.npi_pattern.match(clean_npi):
            result['errors'].append('NPI必须是10位数字')
            return result
        
        # Luhn算法验证（NPI使用Luhn算法）
        if self._luhn_check(clean_npi):
            result['is_valid'] = True
            result['formatted'] = clean_npi
        else:
            result['errors'].append('NPI校验位不正确')
        
        return result
    
    def _luhn_check(self, number: str) -> bool:
        """Luhn算法校验"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0
    
    def validate_diagnosis_code(self, code: str) -> Dict[str, any]:
        """验证诊断代码"""
        result = {
            'is_valid': False,
            'formatted': None,
            'code_type': None,
            'errors': []
        }
        
        clean_code = code.strip().upper()
        
        # 检查ICD-10格式
        if self.icd10_pattern.match(clean_code):
            result['is_valid'] = True
            result['formatted'] = clean_code
            result['code_type'] = 'ICD-10'
        else:
            # 检查是否可能是ICD-9或其他格式
            if re.match(r'^\d{3}(\.\d{1,2})?$', clean_code):
                result['code_type'] = 'ICD-9'
                result['formatted'] = clean_code
                result['is_valid'] = True
            else:
                result['errors'].append('诊断代码格式不符合ICD-10或ICD-9标准')
        
        return result
    
    def validate_date(self, date_str: str) -> Dict[str, any]:
        """验证日期格式"""
        result = {
            'is_valid': False,
            'formatted': None,
            'parsed_date': None,
            'errors': []
        }
        
        # 常见日期格式
        date_formats = [
            '%m/%d/%Y',   # MM/DD/YYYY
            '%Y-%m-%d',   # YYYY-MM-DD
            '%m-%d-%Y',   # MM-DD-YYYY
            '%d/%m/%Y',   # DD/MM/YYYY
            '%Y/%m/%d',   # YYYY/MM/DD
            '%m/%d/%y',   # MM/DD/YY
            '%m-%d-%y',   # MM-DD-YY
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.datetime.strptime(date_str.strip(), fmt)
                result['is_valid'] = True
                result['parsed_date'] = parsed_date
                result['formatted'] = parsed_date.strftime('%m/%d/%Y')
                
                # 检查日期合理性
                current_year = datetime.datetime.now().year
                if parsed_date.year < 1900 or parsed_date.year > current_year + 1:
                    result['errors'].append(f'日期年份 {parsed_date.year} 不在合理范围内')
                
                break
            except ValueError:
                continue
        
        if not result['is_valid']:
            result['errors'].append('无法识别的日期格式')
        
        return result
    
    def validate_amount(self, amount_str: str) -> Dict[str, any]:
        """验证金额格式"""
        result = {
            'is_valid': False,
            'formatted': None,
            'numeric_value': None,
            'errors': []
        }
        
        # 清理金额字符串
        clean_amount = re.sub(r'[^\d.,]', '', amount_str)
        
        # 尝试解析金额
        try:
            # 处理逗号分隔符
            if ',' in clean_amount and '.' in clean_amount:
                # 假设最后一个点是小数点
                parts = clean_amount.split('.')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    integer_part = parts[0].replace(',', '')
                    decimal_part = parts[1]
                    numeric_value = float(f"{integer_part}.{decimal_part}")
                else:
                    numeric_value = float(clean_amount.replace(',', ''))
            else:
                numeric_value = float(clean_amount.replace(',', ''))
            
            if numeric_value < 0:
                result['errors'].append('金额不能为负数')
            elif numeric_value > 1000000:
                result['errors'].append('金额过大，请检查')
            else:
                result['is_valid'] = True
                result['numeric_value'] = numeric_value
                result['formatted'] = f"${numeric_value:,.2f}"
                
        except ValueError:
            result['errors'].append('无法解析金额格式')
        except Exception as e:
            result['errors'].append(f'金额验证异常: {e}')
        
        return result
    
    def validate_patient_id(self, patient_id: str) -> Dict[str, any]:
        """验证患者ID"""
        result = {
            'is_valid': False,
            'formatted': None,
            'errors': []
        }
        
        clean_id = patient_id.strip()
        
        # 基本格式检查
        if len(clean_id) < 3:
            result['errors'].append('患者ID长度太短')
        elif len(clean_id) > 20:
            result['errors'].append('患者ID长度太长')
        elif not re.match(r'^[A-Za-z0-9\-_]+$', clean_id):
            result['errors'].append('患者ID包含无效字符')
        else:
            result['is_valid'] = True
            result['formatted'] = clean_id.upper()
        
        return result
    
    def validate_provider_name(self, name: str) -> Dict[str, any]:
        """验证医疗提供者姓名"""
        result = {
            'is_valid': False,
            'formatted': None,
            'name_type': None,
            'errors': []
        }
        
        clean_name = name.strip()
        
        if len(clean_name) < 2:
            result['errors'].append('医疗提供者姓名太短')
            return result
        
        # 检查是否包含医生头衔
        if re.match(r'^(Dr\.?|Doctor)\s+', clean_name, re.IGNORECASE):
            result['name_type'] = 'physician'
        elif any(word in clean_name.lower() for word in ['clinic', 'hospital', 'center', 'medical']):
            result['name_type'] = 'facility'
        else:
            result['name_type'] = 'individual'
        
        # 基本格式检查
        if re.match(r'^[A-Za-z\s\.\-,]+$', clean_name):
            result['is_valid'] = True
            result['formatted'] = clean_name.title()
        else:
            result['errors'].append('医疗提供者姓名包含无效字符')
        
        return result
    
    def validate_all_fields(self, extracted_data: Dict[str, str]) -> Dict[str, Dict]:
        """验证所有抽取的字段"""
        validation_results = {}
        
        # 定义字段验证映射
        field_validators = {
            'provider_name': self.validate_provider_name,
            'provider_phone': self.validate_phone_number,
            'provider_npi': self.validate_npi,
            'patient_id': self.validate_patient_id,
            'patient_name': self.validate_provider_name,  # 使用相同的姓名验证
            'charge_total': self.validate_amount,
            'diagnosis_code': self.validate_diagnosis_code,
            'service_date': self.validate_date
        }
        
        for field_name, field_value in extracted_data.items():
            if field_name in field_validators and field_value:
                validator = field_validators[field_name]
                validation_results[field_name] = validator(field_value)
            else:
                # 未定义验证器的字段，进行基本检查
                validation_results[field_name] = {
                    'is_valid': bool(field_value and field_value.strip()),
                    'formatted': field_value.strip() if field_value else None,
                    'errors': [] if field_value and field_value.strip() else ['字段为空']
                }
        
        return validation_results
    
    def generate_validation_summary(self, validation_results: Dict[str, Dict]) -> Dict:
        """生成验证摘要"""
        summary = {
            'total_fields': len(validation_results),
            'valid_fields': 0,
            'invalid_fields': 0,
            'fields_with_errors': [],
            'overall_score': 0.0,
            'recommendations': []
        }
        
        for field_name, result in validation_results.items():
            if result['is_valid']:
                summary['valid_fields'] += 1
            else:
                summary['invalid_fields'] += 1
                summary['fields_with_errors'].append({
                    'field': field_name,
                    'errors': result['errors']
                })
        
        # 计算总体评分
        if summary['total_fields'] > 0:
            summary['overall_score'] = summary['valid_fields'] / summary['total_fields']
        
        # 生成建议
        if summary['invalid_fields'] > 0:
            summary['recommendations'].append("请检查并更正标记为无效的字段")
        
        if summary['overall_score'] < 0.8:
            summary['recommendations'].append("数据质量较低，建议人工审核")
        elif summary['overall_score'] >= 0.95:
            summary['recommendations'].append("数据质量良好，可直接使用")
        else:
            summary['recommendations'].append("数据质量中等，建议抽样检查")
        
        return summary

def main():
    """测试验证器"""
    validator = MedicalFieldValidator()
    
    # 测试数据
    test_data = {
        'provider_name': 'Dr. John Smith',
        'provider_phone': '(555) 123-4567',
        'provider_npi': '1234567893',  # 示例NPI
        'patient_id': 'P123456',
        'patient_name': 'Jane Doe',
        'charge_total': '$250.00',
        'diagnosis_code': 'M79.3',
        'service_date': '01/15/2024'
    }
    
    print("=== 医疗字段验证测试 ===")
    results = validator.validate_all_fields(test_data)
    
    for field, result in results.items():
        print(f"\n{field}:")
        print(f"  有效: {result['is_valid']}")
        print(f"  格式化: {result['formatted']}")
        if result['errors']:
            print(f"  错误: {result['errors']}")
    
    # 生成摘要
    summary = validator.generate_validation_summary(results)
    print("\n=== 验证摘要 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 