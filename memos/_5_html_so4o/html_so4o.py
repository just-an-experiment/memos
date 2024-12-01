from pydantic import BaseModel, Field
from openai import OpenAI
import os
from bs4 import BeautifulSoup
import cssutils
import logging
from typing import Tuple, List
import re
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor

# Suppress cssutils logging
cssutils.log.setLevel(logging.CRITICAL)

client = OpenAI()

class HTMLOutput(BaseModel):
    html: str = Field(..., description="The generated HTML code")
    css: str = Field(..., description="The CSS styles for the HTML")
    description: str = Field(..., description="Description of the generated HTML")

class HTMLVerifier:
    """Efficient HTML/CSS verification with parallel checks"""
    
    @staticmethod
    def check_html_structure(soup: BeautifulSoup) -> List[str]:
        """Verify HTML structure and semantics"""
        errors = []
        
        # Essential tags check
        essential_tags = {
            'head': 'Missing <head> tag',
            'body': 'Missing <body> tag',
            'title': 'Missing <title> tag'
        }
        for tag, error in essential_tags.items():
            if not soup.find(tag):
                errors.append(error)

        # Check for viewport meta
        if not soup.find('meta', attrs={'name': 'viewport'}):
            errors.append('Missing viewport meta tag')

        # Check for proper heading hierarchy
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            levels = [int(h.name[1]) for h in headings]
            if levels[0] != 1:
                errors.append('First heading is not h1')
            for i in range(1, len(levels)):
                if levels[i] > levels[i-1] + 1:
                    errors.append(f'Improper heading hierarchy: h{levels[i-1]} to h{levels[i]}')

        return errors

    @staticmethod
    def check_accessibility(soup: BeautifulSoup) -> List[str]:
        """Verify accessibility requirements"""
        errors = []
        
        # Check images for alt text
        for img in soup.find_all('img'):
            if not img.get('alt'):
                errors.append(f'Image missing alt text: {img}')

        # Check form inputs for labels
        for input_tag in soup.find_all('input'):
            if input_tag.get('type') not in ['submit', 'button', 'hidden']:
                input_id = input_tag.get('id')
                if input_id:
                    if not soup.find('label', attrs={'for': input_id}):
                        errors.append(f'Input missing associated label: {input_tag}')
                else:
                    errors.append(f'Input missing id attribute: {input_tag}')

        # Check ARIA attributes
        for elem in soup.find_all(attrs={'role': True}):
            role = elem.get('role')
            if role in ['button', 'link'] and not elem.get('aria-label'):
                errors.append(f'Element with role={role} missing aria-label')

        return errors

    @staticmethod
    def check_css_validity(css: str, soup: BeautifulSoup) -> List[str]:
        """Verify CSS validity and usage"""
        errors = []
        
        try:
            sheet = cssutils.parseString(css)
            
            # Track used selectors
            used_selectors = set()
            
            for rule in sheet:
                if hasattr(rule, 'selectorText'):
                    selector = rule.selectorText
                    used_selectors.add(selector)
                    
                    # Skip special selectors
                    if selector in ['*', ':root'] or ':' in selector:
                        continue
                        
                    # Check if selector matches elements
                    try:
                        if not soup.select(selector):
                            errors.append(f'Unused CSS selector: {selector}')
                    except Exception:
                        errors.append(f'Invalid CSS selector: {selector}')

            # Check for unused classes in HTML
            html_classes = set()
            for tag in soup.find_all(class_=True):
                html_classes.update(tag.get('class', []))

            css_classes = set()
            for selector in used_selectors:
                class_matches = re.findall(r'\.([\w-]+)', selector)
                css_classes.update(class_matches)

            unused_classes = html_classes - css_classes
            if unused_classes:
                errors.append(f'Unused HTML classes: {", ".join(unused_classes)}')

        except Exception as e:
            errors.append(f'CSS parsing error: {str(e)}')

        return errors

    @staticmethod
    def check_security(soup: BeautifulSoup) -> List[str]:
        """Check for security issues"""
        errors = []
        
        # Check for inline scripts
        if soup.find_all('script', string=True):
            errors.append('Contains inline JavaScript')

        # Check for inline styles
        if soup.find_all(style=True):
            errors.append('Contains inline styles')

        # Check form security
        for form in soup.find_all('form'):
            if not form.get('action'):
                errors.append('Form missing action attribute')
            if not form.get('method'):
                errors.append('Form missing method attribute')

        return errors

    @classmethod
    def verify(cls, html: str, css: str) -> Tuple[bool, List[str]]:
        """
        Parallel verification of HTML and CSS.
        Returns (is_valid, list_of_errors)
        """
        try:
            # Parse HTML with html5lib for stricter validation
            soup = BeautifulSoup(html, 'html5lib')
            
            # Run checks in parallel
            with ThreadPoolExecutor() as executor:
                future_structure = executor.submit(cls.check_html_structure, soup)
                future_accessibility = executor.submit(cls.check_accessibility, soup)
                future_css = executor.submit(cls.check_css_validity, css, soup)
                future_security = executor.submit(cls.check_security, soup)

            # Collect all errors
            all_errors = []
            all_errors.extend(future_structure.result())
            all_errors.extend(future_accessibility.result())
            all_errors.extend(future_css.result())
            all_errors.extend(future_security.result())

            return len(all_errors) == 0, all_errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

class HTMLFixStrategy:
    """Strategies for fixing specific HTML/CSS issues"""
    
    @staticmethod
    def fix_structure(html: str, errors: List[str]) -> str:
        """Fix HTML structure issues"""
        soup = BeautifulSoup(html, 'html5lib')
        
        # Add missing essential tags
        if not soup.head:
            soup.html.insert(0, soup.new_tag('head'))
        if not soup.body:
            soup.html.append(soup.new_tag('body'))
        if not soup.title:
            soup.head.append(soup.new_tag('title'))
            soup.title.string = "Generated Page"
        
        # Add viewport meta if missing
        if not soup.find('meta', attrs={'name': 'viewport'}):
            viewport = soup.new_tag('meta')
            viewport['name'] = 'viewport'
            viewport['content'] = 'width=device-width, initial-scale=1.0'
            soup.head.append(viewport)
        
        # Fix heading hierarchy
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings and not soup.find('h1'):
            headings[0].name = 'h1'
        
        return str(soup)

    @staticmethod
    def fix_accessibility(html: str, errors: List[str]) -> str:
        """Fix accessibility issues"""
        soup = BeautifulSoup(html, 'html5lib')
        
        # Add missing alt text
        for img in soup.find_all('img', alt=''):
            img['alt'] = f"Image of {img.get('src', '').split('/')[-1].split('.')[0]}"
        
        # Fix form inputs
        for input_tag in soup.find_all('input'):
            if input_tag.get('type') not in ['submit', 'button', 'hidden']:
                if not input_tag.get('id'):
                    input_id = f"input-{input_tag.get('name', 'field')}"
                    input_tag['id'] = input_id
                if not soup.find('label', attrs={'for': input_tag['id']}):
                    label = soup.new_tag('label')
                    label['for'] = input_tag['id']
                    label.string = input_tag.get('placeholder', 'Input field')
                    input_tag.insert_before(label)
        
        # Add ARIA labels
        for elem in soup.find_all(attrs={'role': True}):
            if not elem.get('aria-label'):
                elem['aria-label'] = elem.get_text().strip() or elem.get('role')
        
        return str(soup)

    @staticmethod
    def fix_security(html: str, errors: List[str]) -> str:
        """Fix security issues"""
        soup = BeautifulSoup(html, 'html5lib')
        
        # Remove inline scripts
        for script in soup.find_all('script', string=True):
            script.decompose()
        
        # Move inline styles to style tag
        inline_styles = []
        for elem in soup.find_all(style=True):
            inline_styles.append(elem['style'])
            del elem['style']
        
        if inline_styles:
            style_tag = soup.new_tag('style')
            style_tag.string = '\n'.join([f".generated-style-{i} {{ {style} }}" 
                                        for i, style in enumerate(inline_styles)])
            soup.head.append(style_tag)
        
        # Fix forms
        for form in soup.find_all('form'):
            if not form.get('action'):
                form['action'] = '#'
            if not form.get('method'):
                form['method'] = 'post'
        
        return str(soup)

class GenerationProgress:
    """Progress updates during HTML generation"""
    def __init__(self, step: str, details: str = None):
        self.step = step
        self.details = details

def generate_html(query: str, max_retries: int = 3) -> HTMLOutput:
    system_prompt = """You are an expert HTML/CSS developer. Generate clean, semantic HTML and CSS based on the user's request.

    Essential Requirements:
    1. HTML Structure:
       - Use HTML5 doctype and semantic elements
       - Include <head> with proper meta tags
       - Ensure viewport meta tag for responsiveness
       - Use meaningful <title>
       - Proper heading hierarchy (h1 -> h2 -> h3)

    2. Accessibility:
       - Descriptive alt text for images
       - Proper form labels with for attributes
       - ARIA labels for interactive elements
       - Semantic HTML elements (nav, main, article, etc.)
       - Color contrast compliance

    3. CSS Best Practices:
       - Mobile-first responsive design
       - Flexbox/Grid for layouts
       - BEM naming convention
       - No !important unless necessary
       - Logical property organization

    4. Security:
       - No inline JavaScript
       - No inline styles
       - Proper form attributes
       - Safe external resource loading

    5. Performance:
       - Minimal CSS specificity
       - Efficient selectors
       - Reusable classes
       - Logical component structure

    Add helpful comments to explain complex sections."""

    last_error = None
    fix_strategies = {
        'structure': HTMLFixStrategy.fix_structure,
        'accessibility': HTMLFixStrategy.fix_accessibility,
        'security': HTMLFixStrategy.fix_security
    }

    original_description = None
    
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate HTML and CSS for: {query}"}
                ],
                response_format=HTMLOutput,
            )
            
            if completion.choices[0].message.refusal:
                raise ValueError(f"Model refused to generate HTML: {completion.choices[0].message.refusal}")
            
            result = completion.choices[0].message.parsed
            
            # Store the original description on first attempt
            if attempt == 0:
                original_description = result.description
            
            # Verify and fix specific issues
            is_valid, errors = HTMLVerifier.verify(result.html, result.css)
            if not is_valid:
                # Apply fix strategies to the HTML
                fixed_html = result.html
                for error in errors:
                    for error_type, fix_strategy in fix_strategies.items():
                        if error_type in error.lower():
                            fixed_html = fix_strategy(fixed_html, errors)
                            break
                
                if attempt < max_retries - 1:
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"""
                            Fix these specific issues in the HTML/CSS:
                            {chr(10).join(errors)}
                            
                            Current HTML:
                            {fixed_html}
                            
                            Current CSS:
                            {result.css}
                            
                            Maintain all working elements while fixing the issues.
                            """}
                        ],
                        response_format=HTMLOutput,
                    )
                    
                    new_result = completion.choices[0].message.parsed
                    # Keep the original description
                    new_result.description = original_description
                    result = new_result
                    
                    is_valid, new_errors = HTMLVerifier.verify(result.html, result.css)
                    if is_valid or len(new_errors) < len(errors):
                        if is_valid:
                            return result
                        errors = new_errors
                        continue
                
                raise ValueError(f"Failed to generate valid HTML/CSS after {attempt + 1} attempts. Errors: {chr(10).join(errors)}")
            
            # Keep the original description in the final result
            if attempt > 0:
                result.description = original_description
            return result
                
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate HTML after {max_retries} attempts. Last error: {str(last_error)}")
            continue
    
    raise RuntimeError(f"Failed to generate HTML after {max_retries} attempts. Last error: {str(last_error)}")

def main(query: str, max_retries: int = 3) -> HTMLOutput:
    return generate_html(query, max_retries)

if __name__ == "__main__":
    user_query = input("Enter your request for HTML generation: ")
    result = main(user_query)
    print("\nGenerated HTML:")
    print(result.html)
    print("\nGenerated CSS:")
    print(result.css)
    print("\nDescription:")
    print(result.description)

export = {
    "default": main,
    "generate_html": generate_html,
    "verify_html_css": HTMLVerifier.verify
}
