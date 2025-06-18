# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "html2text",
#   "pillow",
#   "google-generativeai",  # pip install google-generativeai
# ]
# ///

import requests
import json
import os
import re
from pathlib import Path
from datetime import datetime
import html2text
from PIL import Image
from urllib.parse import urljoin, urlparse
import time
 


def load_cookies():
    """Load cookies from cookies.txt file"""
    try:
        with open("cookies.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: cookies.txt file not found!")
        return None


def fetch_discourse_data(cookies, max_pages=10):
    headers = {
        "cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    search_url = "https://discourse.onlinedegree.iitm.ac.in/search.json"
    params = {
        "q": "#courses:tds-kb after:2025-01-01 before:2025-04-15 order:latest"
    }
    all_posts = []
    for page in range(1, max_pages + 1):
        params['page'] = page
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            posts = data.get('posts', [])
            if not posts:
                break
            all_posts.extend(posts)
            print(f"Fetched page {page}: {len(posts)} posts")
            if len(posts) < 50:
                break  # Last page
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    return {'posts': all_posts}


def get_topic_details(topic_id, cookies):
    """Get detailed topic information including posts"""
    headers = {
        "cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"
    try:
        response = requests.get(topic_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching topic {topic_id}: {e}")
        return None

class ImageProcessor:
    """Handle image downloading and description generation"""

    def __init__(self, cookies, base_url="https://discourse.onlinedegree.iitm.ac.in"):
        self.cookies = cookies
        self.base_url = base_url
        self.headers = {
            "cookie": cookies,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.image_cache = {}
        self.description_cache = {}
        self.llm_provider = self._setup_llm()

    def _setup_llm(self):
        """Setup Gemini for image descriptions"""
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✓ Google Gemini API configured successfully")
        return {
            'type': 'gemini',
            'model': model
        }

    def download_image(self, image_url, images_dir="images"):
        """Download image from URL"""
        try:
            if not image_url.startswith('http'):
                image_url = urljoin(self.base_url, image_url)
            if image_url in self.image_cache:
                return self.image_cache[image_url]
            Path(images_dir).mkdir(exist_ok=True)
            response = requests.get(image_url, headers=self.headers)
            response.raise_for_status()
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = f"image_{hash(image_url) % 10000}.jpg"
            filepath = Path(images_dir) / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            result = {
                'local_path': str(filepath),
                'original_url': image_url,
                'filename': filename
            }
            self.image_cache[image_url] = result
            print(f"Downloaded: {filename}")
            return result
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None

    def generate_image_description(self, image_path):
        """Generate description using Gemini"""
        if not self.llm_provider:
            return f"Image: {os.path.basename(image_path)}"
        if image_path in self.description_cache:
            return self.description_cache[image_path]
        try:
            description = self._describe_with_gemini(image_path)
            self.description_cache[image_path] = description
            return description
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            fallback_desc = f"Image: {os.path.basename(image_path)}"
            self.description_cache[image_path] = fallback_desc
            return fallback_desc

    def _describe_with_gemini(self, image_path):
        """Generate description using Google Gemini"""
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        prompt = (
            "Please provide a detailed description of this image. "
            "Focus on the main content, text if visible, charts/diagrams if present, "
            "and any educational or technical content. Be concise but comprehensive."
        )
        response = self.llm_provider['model'].generate_content([prompt, img])
        return response.text

def should_process_image(img_tag, img_src):
    """
    Determine if an image should be processed based on context and characteristics.
    Returns True for content images, False for UI elements.
    """
    if not img_src:
        return False
    ui_patterns = [
        '/avatars/', '/avatar/', '/user_avatar/', '/badges/', '/emoji/',
        '/plugins/', '/assets/', '/stylesheets/', '/icons/', '/favicon',
        '/logo', '.svg',
    ]
    img_src_lower = img_src.lower()
    if any(pattern in img_src_lower for pattern in ui_patterns):
        return False
    img_classes = img_tag.get('class', [])
    if isinstance(img_classes, str):
        img_classes = img_classes.split()
    ui_classes = [
        'avatar', 'emoji', 'badge', 'icon', 'logo', 'user-avatar',
        'topic-avatar', 'favicon',
    ]
    if any(ui_class in ' '.join(img_classes).lower() for ui_class in ui_classes):
        return False
    width = img_tag.get('width')
    height = img_tag.get('height')
    try:
        if width and int(width) < 32:
            return False
        if height and int(height) < 32:
            return False
    except (ValueError, TypeError):
        pass
    alt_text = (img_tag.get('alt', '') or '').lower()
    ui_alt_patterns = [
        'avatar', 'emoji', 'badge', 'icon', 'logo', 'profile', 'user', ':',
    ]
    if any(pattern in alt_text for pattern in ui_alt_patterns):
        return False
    parent = img_tag.parent
    if parent:
        parent_classes = parent.get('class', [])
        if isinstance(parent_classes, str):
            parent_classes = parent_classes.split()
        ui_parent_classes = [
            'avatar', 'user-info', 'topic-meta', 'post-meta', 'emoji', 'badge-wrapper',
        ]
        if any(ui_class in ' '.join(parent_classes).lower() for ui_class in ui_parent_classes):
            return False
    return True

def html_to_markdown_with_images(html_content, image_processor, images_dir="images"):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    images = soup.find_all('img')
    processed_count = 0
    skipped_count = 0
    for img in images:
        img_src = img.get('src')
        if not should_process_image(img, img_src):
            print(f"Skipping UI image: {img_src}")
            skipped_count += 1
            continue
        print(f"Processing content image: {img_src}")
        processed_count += 1
        image_info = image_processor.download_image(img_src, images_dir)
        if not image_info:
            continue
        description = image_processor.generate_image_description(image_info['local_path'])
        original_alt = img.get('alt', '')
        enhanced_alt = f"{original_alt} - {description}".strip(' - ')
        img['alt'] = enhanced_alt
        img['src'] = image_info['local_path']
        caption = soup.new_tag('p')
        caption.string = f"*Image: {description}*"
        img.insert_after(caption)
    print(f"Image processing summary: {processed_count} processed, {skipped_count} skipped")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0
    return h.handle(str(soup))

def sanitize_filename(filename):
    """Sanitize filename for safe file creation"""
    filename = re.sub(r'[<>:"/\\\\|?*]', '', filename)
    filename = re.sub(r'\\s+', '_', filename)
    return filename[:100]

def save_discourse_json(data, filename="discourse_data.json"):
    """Save raw discourse data as JSON"""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"Saved raw data to {filename}")

def convert_to_markdown(discourse_data, cookies, output_dir="discourse_markdowns"):
    """Convert discourse posts to markdown files with enhanced image handling"""
    if not discourse_data or 'posts' not in discourse_data:
        print("No posts found in discourse data")
        return []
    Path(output_dir).mkdir(exist_ok=True)
    images_dir = Path(output_dir) / "images"
    image_processor = ImageProcessor(cookies)
    markdown_files = []
    topics_processed = set()
    for post in discourse_data.get('posts', []):
        topic_id = post.get('topic_id')
        if topic_id in topics_processed:
            continue
        topics_processed.add(topic_id)
        topic_data = get_topic_details(topic_id, cookies)
        if not topic_data:
            continue
        topic_title = topic_data.get('title', f'Topic_{topic_id}')
        topic_slug = topic_data.get('slug', '')
        markdown_content = f"# {topic_title}\n\n"
        markdown_content += f"**Topic ID:** {topic_id}\n"
        markdown_content += f"**URL:** https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}\n"
        markdown_content += f"**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        if 'details' in topic_data:
            details = topic_data['details']
            if 'created_at' in details:
                markdown_content += f"**Created:** {details['created_at']}\n"
            if 'last_posted_at' in details:
                markdown_content += f"**Last Posted:** {details['last_posted_at']}\n"
            markdown_content += "\n"
        markdown_content += "## Posts\n\n"
        for i, topic_post in enumerate(topic_data.get('post_stream', {}).get('posts', [])):
            post_content = topic_post.get('cooked', '')
            username = topic_post.get('username', 'Unknown')
            created_at = topic_post.get('created_at', '')
            markdown_content += f"### Post {i+1} - {username}\n"
            markdown_content += f"**Posted:** {created_at}\n\n"
            if post_content:
                print(f"Processing post {i+1} content...")
                post_markdown = html_to_markdown_with_images(
                    post_content,
                    image_processor,
                    str(images_dir)
                )
                markdown_content += post_markdown + "\n\n"
            markdown_content += "---\n\n"
        safe_title = sanitize_filename(topic_title)
        filename = f"{safe_title}_{topic_id}.md"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(markdown_content)
        markdown_files.append(str(filepath))
        print(f"Created: {filepath}")
        time.sleep(0.5)
    if images_dir.exists() and list(images_dir.glob("*")):
        create_image_index(images_dir, image_processor.description_cache)
    return markdown_files

def create_image_index(images_dir, description_cache):
    """Create an index of all downloaded images with descriptions"""
    index_content = "# Image Index\n\n"
    index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']]
    for img_file in sorted(image_files):
        img_path = str(img_file)
        description = description_cache.get(img_path, "No description available")
        index_content += f"## {img_file.name}\n\n"
        index_content += f"![{description}]({img_file.name})\n\n"
        index_content += f"**Description:** {description}\n\n"
        index_content += "---\n\n"
    index_path = images_dir / "image_index.md"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write(index_content)
    print(f"Created image index: {index_path}")

def combine_with_existing_markdowns(new_markdowns, existing_dir="markdowns", combined_dir="combined_markdowns"):
    """Combine new discourse markdowns with existing markdown folder"""
    combined_path = Path(combined_dir)
    combined_path.mkdir(exist_ok=True)
    existing_path = Path(existing_dir)
    if existing_path.exists():
        for md_file in existing_path.glob("*.md"):
            dest_file = combined_path / md_file.name
            with open(md_file, "r", encoding="utf-8") as src:
                content = src.read()
            with open(dest_file, "w", encoding="utf-8") as dst:
                dst.write(content)
            print(f"Copied: {md_file.name}")
    for md_file_path in new_markdowns:
        md_file = Path(md_file_path)
        dest_file = combined_path / f"discourse_{md_file.name}"
        with open(md_file, "r", encoding="utf-8") as src:
            content = src.read()
        with open(dest_file, "w", encoding="utf-8") as dst:
            dst.write(content)
        print(f"Added discourse file: discourse_{md_file.name}")
    print(f"\nAll markdowns combined in: {combined_dir}")
    return str(combined_path)

def create_index_file(combined_dir="combined_markdowns"):
    """Create an index file listing all markdown files"""
    combined_path = Path(combined_dir)
    index_content = "# Markdown Files Index\n\n"
    index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_files = list(combined_path.glob("*.md"))
    index_content += "## Discourse Files\n\n"
    discourse_files = [f for f in md_files if f.name.startswith("discourse_")]
    for md_file in sorted(discourse_files):
        index_content += f"- [{md_file.stem}]({md_file.name})\n"
    index_content += "\n## Other Files\n\n"
    other_files = [f for f in md_files if not f.name.startswith("discourse_")]
    for md_file in sorted(other_files):
        index_content += f"- [{md_file.stem}]({md_file.name})\n"
    index_path = combined_path / "README.md"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write(index_content)
    print(f"Created index file: {index_path}")

def main():
    """Main execution function"""
    print("Starting Discourse to Markdown extraction with AI image descriptions...")
    api_configured = bool(os.getenv('GOOGLE_API_KEY'))
    if not api_configured:
        print("\n" + "="*60)
        print("IMAGE DESCRIPTION SETUP")
        print("="*60)
        print("For AI-generated image descriptions, set the environment variable:")
        print("• GOOGLE_API_KEY - for Gemini 2.0 Flash")
        print("\nImages will still be downloaded but won't have AI descriptions.")
        print("="*60 + "\n")
    cookies = load_cookies()
    if not cookies:
        return
    print("Fetching discourse data...")
    discourse_data = fetch_discourse_data(cookies)
    if not discourse_data:
        return
    save_discourse_json(discourse_data)
    print("Converting to markdown with image processing...")
    markdown_files = convert_to_markdown(discourse_data, cookies)
    if not markdown_files:
        print("No markdown files created.")
        return
    print(f"Created {len(markdown_files)} markdown files")
    print("Combining with existing markdowns...")
    combined_dir = combine_with_existing_markdowns(markdown_files)
    create_index_file(combined_dir)
    print("\nProcess completed successfully!")
    print(f"Check the '{combined_dir}' folder for all combined markdown files.")
    print("Images are stored in the 'discourse_markdowns/images/' directory.")
    print("Check 'discourse_markdowns/images/image_index.md' for image descriptions.")

if __name__ == "__main__":
    main()
