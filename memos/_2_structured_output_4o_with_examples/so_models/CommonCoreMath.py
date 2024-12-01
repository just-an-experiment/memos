# A Pydantic model representing a collection of Common Core math topics and their details.

from pydantic import BaseModel, Field

class CommonCoreMathTopic(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the math topic within the Common Core standards.")
    title: str = Field(..., description="Title or name of the math topic.")
    grade_level: str = Field(..., description="Grade level associated with the math topic.")
    description: str = Field(..., description="A detailed description of the math topic.")
    standards: list[str] = Field(..., description="List of relevant Common Core math standards identifiers.")
    skill_list: list[str] = Field(..., description="List of specific skills or competencies associated with the math topic.")
    resources: list[str] = Field(..., description="List of resources or references for teaching the math topic.")

class CommonCoreMath(BaseModel):
    topics: list[CommonCoreMathTopic] = Field(..., description="A list of Common Core Math topics covered, with details.")

# Example data that matches the model schema
examples = [
    {'topics': [{'identifier': 'CCSS.MATH.CONTENT.K.G.A.1', 'title': 'Identify Shapes', 'grade_level': 'Kindergarten', 'description': 'Describe objects in the environment using names of shapes, and describe the relative positions of these objects.', 'standards': ['K.G.A.1'], 'skill_list': ['Identify and name shapes such as rectangles, circles, triangles, squares', "Use positional words like 'above', 'below', 'beside', 'in front of', 'behind', 'next to'"], 'resources': ['https://www.khanacademy.org/math/cc-kindergarten-math/cc-kindergarten-geometry']}, {'identifier': 'CCSS.MATH.CONTENT.3.NF.A.1', 'title': 'Understanding Fractions', 'grade_level': 'Grade 3', 'description': 'Understand a fraction 1/b as the quantity formed by 1 part when a whole is partitioned into b equal parts; understand a fraction a/b as the quantity formed by a parts of size 1/b.', 'standards': ['3.NF.A.1'], 'skill_list': ['Interpret fractions as parts of a whole', 'Partition objects into equal parts', 'Represent fractions on a number line'], 'resources': ['https://www.khanacademy.org/math/cc-third-grade-math/cc-3rd-fractions-topic']}]},
    {'topics': [{'identifier': 'CCSS.MATH.CONTENT.5.NBT.A.1', 'title': 'Understanding Decimal Place Values', 'grade_level': 'Grade 5', 'description': 'Recognize that in a multi-digit number, a digit in one place represents ten times what it represents in the place to its right.', 'standards': ['5.NBT.A.1'], 'skill_list': ['Read and write decimals to thousandths', 'Understand the base-ten place value system', 'Compare and round decimals'], 'resources': ['https://www.khanacademy.org/math/cc-fifth-grade-math/decimals-place-value']}, {'identifier': 'CCSS.MATH.CONTENT.7.EE.B.3', 'title': 'Solving Real-world Problems with Rational Numbers', 'grade_level': 'Grade 7', 'description': 'Solve multi-step real-world and mathematical problems posed with positive and negative rational numbers in any form (whole numbers, fractions, and decimals), using tools strategically.', 'standards': ['7.EE.B.3'], 'skill_list': ['Solve real-world problems with rational numbers', 'Use estimation strategies', 'Apply operations in different contexts'], 'resources': ['https://www.khanacademy.org/math/algebra/arith-review']}]},
    {'topics': [{'identifier': 'CCSS.MATH.CONTENT.8.G.B.6', 'title': 'Understanding Similarity', 'grade_level': 'Grade 8', 'description': 'Explain a proof of the Pythagorean Theorem and its converse.', 'standards': ['8.G.B.6'], 'skill_list': ['Understand relationships between side lengths of right triangles', 'Prove the Pythagorean Theorem', 'Apply the converse of the Pythagorean Theorem'], 'resources': ['https://www.khanacademy.org/math/geometry/hs-geo-foundations']}, {'identifier': 'CCSS.MATH.CONTENT.HSF.LE.A.1', 'title': 'Exponential Models', 'grade_level': 'High School', 'description': 'Distinguish between situations that can be modeled with linear functions and with exponential functions.', 'standards': ['HSF.LE.A.1'], 'skill_list': ['Identify relationships in tables and graphs', 'Compare linear to exponential growth', 'Model real-world situations with functions'], 'resources': ['https://www.khanacademy.org/math/algebra/exponentials-logarithms']}]},
]


export = {
    'default': CommonCoreMath,
    'examples': examples
}