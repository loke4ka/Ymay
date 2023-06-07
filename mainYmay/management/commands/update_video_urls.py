from mainYmay.models import Video


def update_video_urls():
    videos = Video.objects.all()

    for video in videos:
        old_url = video.url
        new_url = get_embed_url(old_url)
        video.url = new_url
        video.save()


def get_embed_url(url):
    # Проверка, если ссылка уже имеет формат embed, то возвращаем без изменений
    if 'embed' in url:
        return url

    # Разделение ссылки на части
    parts = url.split('=')
    video_id = parts[-1]

    # Формирование новой ссылки в формате embed
    embed_url = f'https://www.youtube.com/embed/{video_id}'

    return embed_url


if __name__ == "__main__":
    update_video_urls()
